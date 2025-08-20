#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from collections import defaultdict, namedtuple
import pandas as pd
from lxml import etree as ET
import sqlglot
import os
import glob
import datetime

# -------------------- small utils --------------------

def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag) if isinstance(tag, str) else ""
def lower(s): return (s or "").strip().lower()
def attrs_ci(e): return {k.lower(): (v or "") for k, v in (getattr(e, "attrib", {}) or {}).items()}
def line_no(node):
    ln = getattr(node, "sourceline", None)
    return int(ln) if isinstance(ln, int) else -1
def build_parent_map(root): return {c: p for p in root.iter() for c in p}

def ancestors(e, pm, lim=200):
    cur = e
    for _ in range(lim):
        if cur is None: break
        yield cur
        cur = pm.get(cur)

def dedupe(seq):
    out, seen = [], set()
    for x in seq:
        if x is None: continue
        s = str(x).strip()
        if not s or s in seen: continue
        seen.add(s); out.append(s)
    return out

def _strip_wrappers(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().strip('"').strip("'")
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        s = s[1:-1]
    return s

def _norm_key(s: str) -> str:
    s = _strip_wrappers(s)
    s = re.sub(r"[\{\}\[\]]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"[^A-Z0-9_]", "", s.upper())

def _pretty(s: str) -> str:
    s = _strip_wrappers(s)
    s = re.sub(r"[\{\}\[\]]", "", s)
    return re.sub(r"\s+", " ", s).strip()

class NameBag:
    def __init__(self): self.best = None
    def _score(self, s):
        if not s: return (-1, -1, -1)
        has_space = 1 if " " in s else 0
        clean     = 1 if (s == _pretty(s)) else 0
        return (has_space, clean, len(s))
    def add(self, s):
        s = _pretty(s)
        if not s: return
        if self.best is None or self._score(s) > self._score(self.best):
            self.best = s
    def get(self, fallback=""): return self.best or _pretty(fallback)

# -------------------- constants --------------------

DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")
CALLSTEP_TAGS = ("dicallstep","callstep")
FUNCTION_DEF_TAGS = (
    "discriptfunction","dicustomfunction","difunction","userfunction",
    "diuserfunction","diuserdefinedfunction","scriptfunction"
)

Record = namedtuple(
    "Record",
    [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table",
        "transformation_position","transformation_usage_count",
        "custom_sql_text","source_line",
    ],
)

# -------------------- project/job/df mapping --------------------

def _job_name_from_node(job_node):
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch,"tag",""))) == "diattribute" and lower(ch.attrib.get("name",""))=="job_name":
            v = (ch.attrib.get("value") or "").strip()
            if v: return v
    return (job_node.attrib.get("name") or job_node.attrib.get("displayName") or "").strip()

def collect_df_names(root):
    out=set()
    for n in root.iter():
        if lower(strip_ns(getattr(n,"tag",""))) in DF_TAGS:
            nm = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: out.add(nm)
    return out

def build_job_to_project_map(root):
    j2p={}
    for p in root.iter():
        if lower(strip_ns(getattr(p,"tag",""))) not in PROJECT_TAGS:
            continue
        proj=(p.attrib.get("name") or p.attrib.get("displayName") or "").strip()
        if not proj: continue
        for ref in p.iter():
            if lower(strip_ns(getattr(ref,"tag",""))) == "dijobref":
                jn=(ref.attrib.get("name") or ref.attrib.get("displayName") or "").strip()
                if jn: j2p.setdefault(jn, proj)
    return j2p

def build_df_project_map(root):
    df_names = collect_df_names(root)
    df_proj={}
    projects=[]
    for p in root.iter():
        if lower(strip_ns(getattr(p,"tag",""))) in PROJECT_TAGS:
            nm=(p.attrib.get("name") or p.attrib.get("displayName") or "").strip()
            if nm: projects.append((nm,p))
    for pnm,p in projects:
        for d in p.iter():
            if lower(strip_ns(getattr(d,"tag",""))) in DF_TAGS:
                dn=(d.attrib.get("name") or d.attrib.get("displayName") or "").strip()
                if dn: df_proj.setdefault(dn, pnm)
    if len(projects)==1:
        only=projects[0][0]
        for dn in collect_df_names(root): df_proj.setdefault(dn, only)
    return df_proj

def build_df_job_map(root):
    pm={c:p for p in root.iter() for c in p}
    df_names = collect_df_names(root)
    df_canon = {re.sub(r'[^A-Z0-9]','',n.upper()): n for n in df_names}

    jobs = {}
    wfs  = {}
    for n in root.iter():
        t=lower(strip_ns(getattr(n,"tag","")))
        if t in JOB_TAGS:
            nm=_job_name_from_node(n)
            if nm: jobs[nm]=n
        elif t in WF_TAGS:
            nm=(n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: wfs[nm]=n

    edges=defaultdict(set)
    def canon(s): return re.sub(r'[^A-Z0-9]','',(s or '').upper())
    def add_edge(src_kind, src_name, dst_kind, dst_name):
        if src_name and dst_name:
            edges[(src_kind, canon(src_name))].add((dst_kind, canon(dst_name)))

    for cs in root.iter():
        if lower(strip_ns(getattr(cs,"tag",""))) not in CALLSTEP_TAGS: continue
        # climb to find src job/wf
        cur=cs; src_kind=src_name=None
        for _ in range(200):
            cur=pm.get(cur)
            if cur is None: break
            t=lower(strip_ns(cur.tag))
            if t in JOB_TAGS: src_kind,src_name="job",_job_name_from_node(cur); break
            if t in WF_TAGS:  src_kind,src_name="wf",(cur.attrib.get("name") or cur.attrib.get("displayName") or ""); break
        if not src_name: continue

        a=attrs_ci(cs)
        tgt_type=(a.get("calledobjecttype") or a.get("type") or "").strip().lower()
        names=[]
        for k in ("calledobject","name","object","target","called_object"):
            if a.get(k):
                raw=a.get(k); names.append(raw)
                if any(sep in raw for sep in ["/","\\",".",":"]):
                    names.append(raw.split("/")[-1].split("\\")[-1].split(":")[-1].split(".")[-1])

        txt=" ".join([*a.values()])

        if tgt_type in ("workflow","diworkflow"):
            for nm in names: add_edge(src_kind,src_name,"wf",nm)
        elif tgt_type in ("dataflow","didataflow"):
            for nm in names: add_edge(src_kind,src_name,"df",nm)
        else:
            for w in wfs.keys():
                if canon(w) in canon(txt): add_edge(src_kind,src_name,"wf",w)
            for d in df_names:
                if canon(d) in canon(txt): add_edge(src_kind,src_name,"df",d)

    df_job={}
    for j in jobs.keys():
        start=("job", re.sub(r'[^A-Z0-9]','',j.upper()))
        seen={start}; stack=[start]; reach=set()
        while stack:
            node=stack.pop()
            for nxt in edges.get(node,()):
                if nxt in seen: continue
                seen.add(nxt)
                kind, nm = nxt
                if kind=="df":
                    real=df_canon.get(nm)
                    if real: reach.add(real)
                else:
                    stack.append(nxt)
        for d in reach: df_job.setdefault(d, j)

    if len(jobs)==1:
        only=list(jobs.keys())[0]
        for d in df_names: df_job.setdefault(d, only)
    return df_job

# -------------------- helpers for positions --------------------

def schema_out_from_DISchema(e, pm, fallback=""):
    best=None; join=None
    for a in ancestors(e, pm, 200):
        if lower(strip_ns(getattr(a,"tag","")))=="dischema":
            nm=(attrs_ci(a).get("name") or "").strip()
            if nm:
                if lower(nm)!="join": best=nm; break
                else: join=nm
    return best or join or fallback

def find_output_column(e, pm):
    if lower(strip_ns(getattr(e,"tag","")))=="dielement":
        nm=(attrs_ci(e).get("name") or "").strip()
        if nm: return nm
    cur=e
    for _ in range(200):
        if cur is None: break
        if lower(strip_ns(cur.tag))=="dielement":
            nm=(attrs_ci(cur).get("name") or "").strip()
            if nm: return nm
        cur=pm.get(cur)
    return ""

def in_dataflow(e, pm):
    for a in ancestors(e, pm, 200):
        if lower(strip_ns(getattr(a,"tag",""))) in DF_TAGS:
            return True
    return False

# -------- NEW: robust UI display name finder for custom SQL --------
def find_ui_display_name(e, pm):
    """
    Prefer the nearest DISchema's UI display name; fall back to DISchema name;
    finally scan higher ancestors for UI display attributes.
    """
    # 1) nearest DISchema
    nearest_schema = None
    for up in ancestors(e, pm, 400):
        if lower(strip_ns(getattr(up, "tag", ""))) == "dischema":
            nearest_schema = up
            break

    def find_ui_in(node):
        for ch in node.iter():
            if lower(strip_ns(getattr(ch, "tag", ""))) == "diattribute":
                nm = lower(getattr(ch, "attrib", {}).get("name", ""))
                if nm in ("ui_display_name", "ui_acta_from_schema_0"):
                    val = (getattr(ch, "attrib", {}).get("value") or "").strip()
                    if val:
                        return val
        return ""

    if nearest_schema is not None:
        v = find_ui_in(nearest_schema)
        if v:
            return v
        # fallback to schema tag name
        v = (attrs_ci(nearest_schema).get("name") or "").strip()
        if v:
            return v

    # 2) last resort: search broader ancestors for the attribute
    for up in ancestors(e, pm, 400):
        v = find_ui_in(up)
        if v:
            return v

    return ""

# -------------------- SQL helpers --------------------

def clean_sql(sql_text: str) -> str:
    # Replace parameterized schema like ${G_Schema} or [${G_Schema}]
    sql_text = re.sub(r"\$\{[^}]+\}", "DUMMY_SCHEMA", sql_text)
    sql_text = re.sub(r"\[\$\{[^}]+\}\]", "DUMMY_SCHEMA", sql_text)

    # Remove remaining square brackets
    sql_text = re.sub(r"[\[\]]", "", sql_text)

    # Remove single-line comments (-- ...)
    sql_text = re.sub(r"--.*?$", "", sql_text, flags=re.MULTILINE)

    # Remove multi-line comments (/* ... */)
    sql_text = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.DOTALL)

    # Collapse all newlines / tabs into a single space
    sql_text = re.sub(r"\s+", " ", sql_text).strip()

    # Strip extra whitespace
    sql_text = sql_text.strip()
    return sql_text

def extract_tables_from_sql(sql_text: str):
    """
    Robust table extractor using sqlglot. Falls back to the original regex
    if parsing fails. Returns deduped list of table names, preserving
    schema.table when available.
    """
    if not sql_text:
        return []
    try:
        # Parse the SQL text (handles multi-statement strings)
        expressions = sqlglot.parse(sql_text)
        tables = set()

        # Collect CTE names to avoid counting them as physical tables
        cte_names = set()
        for expr in expressions:
            for cte in expr.find_all(sqlglot.exp.CTE):
                alias = cte.args.get("alias")
                if isinstance(alias, sqlglot.exp.TableAlias):
                    if alias.name:
                        cte_names.add(alias.name.upper())

        # Walk the AST and grab all Table nodes
        for expr in expressions:
            for t in expr.find_all(sqlglot.exp.Table):
                # Skip references that are actually CTE aliases
                t_name = t.name  # unquoted table identifier
                t_db   = t.args.get("db") or ""  # schema / database
                if not t_db and t_name and t_name.upper() in cte_names:
                    continue

                if t_db:
                    tables.add(f"{t_db}.{t_name}")
                else:
                    tables.add(t_name)
        table_list=dedupe(list(tables))
        if (not table_list or any(t.endswith('.'))):raise ValueError('Invalid Table')
        return table_list

    except Exception:
        # Parser couldn’t handle it -> fallback to your proven regex logic
        table_list=extract_tables_from_sql_regex(sql_text)
        return table_list

def extract_tables_from_sql_regex(sql_text: str):
    SQL_FROM_JOIN_RE = re.compile(r"\b(?:from|join)\s+([A-Za-z0-9_\.\$#@]+)", re.I)
    c = " ".join(sql_text.replace("\n", " ").replace("\r", " ").split())
    hits = SQL_FROM_JOIN_RE.findall(c)
    tables = []
    for h in hits:
        parts = h.split(".")
        if len(parts) >= 2:
            tables.append(f"{parts[-2]}.{parts[-1]}")
        else:
            tables.append(parts[-1])
    return dedupe(tables)

# -------------------- main parser --------------------

def parse_single_xml(xml_path: str) -> pd.DataFrame:
    parser = ET.XMLParser(huge_tree=True, recover=True)
    tree   = ET.parse(xml_path, parser=parser)
    root   = tree.getroot()
    pm     = build_parent_map(root)

    job_to_project = build_job_to_project_map(root)
    df_to_project  = build_df_project_map(root)
    df_to_job      = build_df_job_map(root)

    display_ds  = defaultdict(NameBag)
    display_sch = defaultdict(NameBag)
    display_tbl = defaultdict(NameBag)
    def remember_display(ds, sch, tbl):
        ds=_strip_wrappers(ds).strip()
        sch=_strip_wrappers(sch).strip()
        tbl=_strip_wrappers(tbl).strip()
        k=(_norm_key(ds), _norm_key(sch), _norm_key(tbl))
        display_ds[k].add(ds); display_sch[k].add(sch); display_tbl[k].add(tbl)

    # NOTE: now include source_line with each collected item
    source_target  = set()  # (proj,job,df,role,dsN,schN,tblN, line)
    sql_rows       = []
    lookup_pos     = defaultdict(lambda: defaultdict(set))   # key -> pos -> {call_lines}
    lookup_ext_pos = defaultdict(lambda: defaultdict(set))   # key -> pos -> {call_lines}
    missing_lookup = []

    df_context        = {}  # df -> (proj, job)
    df_func_callsites = defaultdict(lambda: defaultdict(set))  # df -> canon(func) -> {(pos, call_line)}

    cur_proj = cur_job = cur_df = cur_schema = ""
    last_job = ""

    def context_for(e):
        proj=job=df=""
        for a in ancestors(e, pm):
            t  = lower(strip_ns(a.tag))
            at = attrs_ci(a)
            nm = (at.get("name") or at.get("displayname") or "").strip()
            if not df and t in DF_TAGS: df = nm or df
            if not proj and t in PROJECT_TAGS: proj = nm or proj
            if t in JOB_TAGS and not job: job = _job_name_from_node(a) or job
        df  = df or cur_df
        job = job or df_to_job.get(df, None) or (last_job if not cur_job else cur_job)
        proj = job_to_project.get(job, proj or df_to_project.get(df, cur_proj))
        if df and (df not in df_context): df_context[df]=(proj or "", job or "")
        return proj or "", job or "", df or ""

    def normalize_fn_name(name: str) -> str:
        if not name: return ""
        s = name.strip()
        for sep in ("::","/","."):
            if sep in s: s=s.split(sep)[-1]
        return re.sub(r'[^A-Z0-9]','',s.upper())

    # collect lookup calls inside function definitions (for later attribution)
    fn_lookup_calls = defaultdict(list)   # canon(func) -> list of (role, ds, sch, tbl)
    for node in root.iter():
        t=lower(strip_ns(getattr(node,"tag","")))
        if t in FUNCTION_DEF_TAGS:
            nm=(getattr(node,"attrib",{}) or {}).get("name") or (getattr(node,"attrib",{}) or {}).get("displayName") or ""
            key = normalize_fn_name(nm)
            if not key: continue
            for fc in node.iter():
                if lower(strip_ns(getattr(fc,"tag","")))!="function_call": continue
                an = attrs_ci(fc)
                cal = (an.get("name") or "").strip().lower()
                if cal not in ("lookup","lookup_ext"): continue
                role = "lookup_ext" if cal=="lookup_ext" else "lookup"
                ds   = an.get("tabledatastore") or ""
                sch  = an.get("tableowner") or ""
                tbl  = an.get("tablename") or ""
                fn_lookup_calls[key].append((role, ds, sch, tbl))

    # walk xml (only DF-resident things are recorded)
    for e in root.iter():
        if not isinstance(e.tag, str): continue
        tag=lower(strip_ns(e.tag)); a=attrs_ci(e)

        if tag in PROJECT_TAGS: cur_proj=(a.get("name") or a.get("displayname") or cur_proj).strip()
        if tag in DF_TAGS:
            cur_df  =(a.get("name") or a.get("displayname") or cur_df).strip()
            if cur_df and cur_df not in df_context:
                p,j,_=context_for(e); df_context[cur_df]=(p,j)
        if tag in JOB_TAGS:
            cur_job=_job_name_from_node(e) or (a.get("name") or a.get("displayname") or cur_job).strip()
            if cur_job: last_job=cur_job
        if tag=="dischema":
            cur_schema=(a.get("name") or a.get("displayname") or cur_schema).strip()

        # 1) DB sources/targets (only if inside DF) — now with source_line
        if tag in ("didatabasetablesource","didatabasetabletarget") and in_dataflow(e, pm):
            proj,job,df=context_for(e)
            role="source" if "source" in tag else "target"
            ds =(a.get("datastorename") or a.get("datastore") or "").strip()
            sch=(a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl=(a.get("tablename") or a.get("table") or "").strip()
            if ds and tbl:
                remember_display(ds,sch,tbl)
                key=(proj,job,df,role,_norm_key(ds),_norm_key(sch),_norm_key(tbl), line_no(e))
                source_target.add(key)

        # 2) FILE / EXCEL sources/targets (only if inside DF) — now with source_line
        if tag in ("difilesource","difiletarget","diexcelsource","diexceltarget") and in_dataflow(e, pm):
            proj,job,df = context_for(e)
        
            # role from tag (Excel "source" should always be source)
            role = "source" if ("source" in tag) else "target"
        
            a = attrs_ci(e)
        
            # Try to capture some notion of format
            fmt = (a.get("formatname") or a.get("file_format") or "").strip()
        
            # Prefer inner DIOutputView name (often the sheet/logical name) for "table"
            outview_name = ""
            for ch in e.iter():
                if lower(strip_ns(getattr(ch, "tag", ""))) in ("dioutputview","outputview"):
                    outview_name = (attrs_ci(ch).get("name") or "").strip()
                    if outview_name:
                        break
        
            # Filename / dataset name / element name
            fname = (
                a.get("filename")     or
                a.get("name")         or
                a.get("datasetname")  or
                ""
            ).strip()
            if outview_name:
                # if we found a nicer logical name, prefer it
                fname = fname or outview_name
        
            if tag in ("diexcelsource","diexceltarget"):
                ds  = (a.get("datastorename") or a.get("datastore") or "EXCEL").strip()
                sch = (fmt or "EXCEL")
                tbl = (fname or "EXCEL_OBJECT")
            else:
                ds  = (a.get("datastorename") or a.get("datastore") or "FILE").strip() or "FILE"
                sch = (fmt or "FILE")
                tbl = (fname or "FILE_OBJECT")
        
            remember_display(ds, sch, tbl)
            key = (proj, job, df, role, _norm_key(ds), _norm_key(sch), _norm_key(tbl), line_no(e))
            source_target.add(key)

        # 3) CUSTOM SQL (inside DF)
        if tag in ("sqltext","sqltexts","diquery","ditransformcall") and in_dataflow(e, pm):
            sql_text=""
            if tag in ("sqltext","sqltexts"):
                sql_text=(a.get("sql_text") or "").strip() or (e.text or "").strip()
            if not sql_text:
                for ch in e.iter():
                    if lower(strip_ns(getattr(ch,"tag",""))) in ("sqltext","sql_text"):
                        sql_text=(attrs_ci(ch).get("sql_text") or ch.text or "").strip()
                        if sql_text: break
            if sql_text:
                proj,job,df=context_for(e)

                # NEW: robust UI display name -> goes to SCHEMA
                disp_name = find_ui_display_name(e, pm)

                # tables from SQL -> TABLE
                sql_text_clean=clean_sql(sql_text)
                tables=extract_tables_from_sql(sql_text_clean)
                table_csv=", ".join(sorted(tables)) if tables else "NO_SCHEMA.NO_TABLE"

                # datastore: read the real database_datastore from ancestors (fallback to DS_SQL)
                ds_for_sql=""
                for up in ancestors(e, pm, 12):
                    for ch in up.iter():
                        if lower(strip_ns(getattr(ch,"tag","")))=="diattribute" and lower(getattr(ch, "attrib", {}).get("name",""))=="database_datastore":
                            ds_for_sql=(getattr(ch, "attrib", {}).get("value") or "").strip()
                            if ds_for_sql: break
                    if ds_for_sql: break
                ds_for_sql = ds_for_sql or "NO_DS"

                # remember pretty names
                remember_display(ds_for_sql, disp_name, table_csv)

                # transformation_position must be blank for custom SQL
                sql_rows.append(Record(
                    proj,job,df,"custom_sql",
                    ds_for_sql,           # datastore
                    disp_name,            # schema (UI display name)
                    table_csv,            # table (comma-separated tables)
                    disp_name,                   # transformation_position BLANK
                    len(tables),
                    '"' + (sql_text.replace('"','""')) + '"',
                    line_no(e),
                ))

        # 4) LOOKUPS — FUNCTION_CALL inside DF (unchanged from your v12)
        if tag=="function_call" and in_dataflow(e, pm):
            proj,job,df = context_for(e)
            an  = attrs_ci(e)
            cal = (an.get("name") or "").strip().lower()

            if cal not in ("lookup","lookup_ext"):
                schema_out = schema_out_from_DISchema(e, pm, cur_schema)
                col        = find_output_column(e, pm)
                if schema_out:
                    pos = f"{schema_out}>>{col}" if col else schema_out
                    fn_key = normalize_fn_name(an.get("name") or "")
                    if fn_key: df_func_callsites[df][fn_key].add((pos, line_no(e)))
                continue

            schema_out = schema_out_from_DISchema(e, pm, cur_schema)
            col        = find_output_column(e, pm)
            pos        = f"{schema_out}>>{col}" if (schema_out and col) else (schema_out or "")
            ds  = an.get("tabledatastore") or ""
            sch = an.get("tableowner") or ""
            tbl = an.get("tablename") or ""
            ln  = line_no(e)
            role = "lookup_ext" if cal=="lookup_ext" else "lookup"

            if not (ds and tbl):
                missing_lookup.append(Record(
                    proj, job, df, role,
                    ds or "<missing>", sch or "<missing>", tbl or "<missing>",
                    pos, 1, "", ln
                ))
            else:
                remember_display(ds, sch, tbl)
                key = (proj, job, df, _norm_key(ds), _norm_key(sch), _norm_key(tbl))
                if role == "lookup_ext":
                    lookup_ext_pos[key][pos].add(ln)
                else:
                    lookup_pos[key][pos].add(ln)

    # 5) expand external function lookups ONLY for DFs that actually call them
    for df_name, fn_map in df_func_callsites.items():
        proj,job = df_context.get(df_name, ("",""))
        if not proj:
            inferred_job = df_to_job.get(df_name, "")
            job = job or inferred_job
            proj = job_to_project.get(job, "") or df_to_project.get(df_name, "")
        for fn_key, callsites in fn_map.items():
            payloads = fn_lookup_calls.get(fn_key, [])
            if not payloads:  # function doesn't contain lookups; skip
                continue
            for role, ds, sch, tbl in payloads:
                if not (ds and tbl):
                    for pos, ln_call in callsites:
                        missing_lookup.append(Record(
                            proj, job, df_name, role,
                            ds or "<missing>", sch or "<missing>", tbl or "<missing>",
                            pos, 1, "", ln_call
                        ))
                    continue
                remember_display(ds, sch, tbl)
                key = (proj, job, df_name, _norm_key(ds), _norm_key(sch), _norm_key(tbl))
                if role == "lookup_ext":
                    for pos, ln_call in callsites: lookup_ext_pos[key][pos].add(ln_call)
                else:
                    for pos, ln_call in callsites: lookup_pos[key][pos].add(ln_call)

    # -------------------- finalize --------------------

    def nice_names(dsN, schN, tblN):
        k=(dsN, schN, tblN)
        return (display_ds[k].get(dsN), display_sch[k].get(schN), display_tbl[k].get(tblN))

    rows=[]

    for (proj,job,df,dsN,schN,tblN), posmap in lookup_pos.items():
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        positions=[]; lines=[]
        for p, lnset in posmap.items():
            if p: positions.append(p)
            lines.extend(sorted(lnset))
        positions = sorted(dedupe(positions))
        rows.append(Record(proj, job, df, "lookup", dsD, schD, tblD,
                           ", ".join(positions), len(positions), "", min(lines) if lines else -1))

    for (proj,job,df,dsN,schN,tblN), posmap in lookup_ext_pos.items():
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        positions=[]; lines=[]
        for p, lnset in posmap.items():
            if p: positions.append(p)
            lines.extend(sorted(lnset))
        positions = sorted(dedupe(positions))
        rows.append(Record(proj, job, df, "lookup_ext", dsD, schD, tblD,
                           ", ".join(positions), len(positions), "", min(lines) if lines else -1))

    rows.extend(missing_lookup)

    # include source_line from source/target tuples
    for (proj,job,df,role,dsN,schN,tblN,ln) in sorted(source_target):
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        rows.append(Record(proj, job, df, role, dsD, schD, tblD, "", 0, "", ln))

    rows.extend(sql_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def fill_proj(row):
        if str(row["project_name"]).strip(): return row["project_name"]
        j = row.get("job_name",""); d = row.get("dataflow_name","")
        if j and j in job_to_project: return job_to_project[j]
        return df_to_project.get(d, "")

    df["project_name"] = df.apply(fill_proj, axis=1)

    def nkey(r):
        return (
            r["project_name"], r["job_name"], r["dataflow_name"], r["role"],
            _norm_key(r["datastore"]), _norm_key(r["schema"]), _norm_key(r["table"]),
            _norm_key(r.get("custom_sql_text","")), _norm_key(r.get("transformation_position",""))
        )
    df["__k__"]=df.apply(nkey, axis=1)

    df=(df.groupby(
            ["__k__","project_name","job_name","dataflow_name","role",
             "datastore","schema","table","custom_sql_text","transformation_position"],
            dropna=False, as_index=False)
         .agg({"transformation_usage_count": "sum",
               "source_line": lambda x:", ".join(str(i) for i in sorted(set(x)))
              }))

    df=df.drop(columns=["__k__"])

    for c in ("datastore","schema","table","custom_sql_text","transformation_position"):
        df[c]=df[c].map(_pretty)

    df=df.sort_values(by=[
        "project_name","job_name","dataflow_name","role","datastore","schema","table","transformation_position"
    ]).reset_index(drop=True)
    return df

# -------------------- main --------------------

def main():
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")

    # keep your paths here
    path = r"C:\Users\raksahu\Downloads\python\input\sap_ds_xml_files"
    single_file = fr"{path}\export_af.xml"

    all_files = glob.glob(os.path.join(path, "*.xml"))
    print(all_files)
    if single_file in all_files:
        all_files = [single_file]

    print(f"total number of files present in the path ({len(all_files)})")
    print(all_files)

    df_list = []
    for i, file in enumerate(all_files):
        print(f"Row Number:{i}--{file}")
        df = parse_single_xml(file)
        df_list.append(df)

    final_df = pd.concat(df_list, ignore_index=True)

    rename_mapping = {
        'project_name'               : 'PROJECT_NAME',
        'job_name'                   : 'JOB_NAME',
        'dataflow_name'              : 'DATAFLOW_NAME',
        'role'                       : 'TRANFORMATION_TYPE',
        'datastore'                  : 'DATA_STORE',
        'schema'                     : 'SCHEMA_NAME',
        'table'                      : 'TABLE_NAME',
        'transformation_position'    : 'TRANSFORMATION_POSITION',
        'transformation_usage_count' : 'TRANSFORMATION_USAGE_COUNT',
        'source_line'                : 'SOURCE_LINE',
        'custom_sql_text'            : 'CUSTOM_SQL_TEXT',
    }

    # keep only these columns, in this order
    final_df = final_df.rename(columns=rename_mapping)[list(rename_mapping.values())]

    # --- keep your key at the end (first 7 columns) ---
    key_cols = ["PROJECT_NAME","JOB_NAME","DATAFLOW_NAME","TRANFORMATION_TYPE","DATA_STORE","SCHEMA_NAME","TABLE_NAME"]
    final_df["RECORD_KEY"] = final_df[key_cols].astype(str).agg("||".join, axis=1)
    final_df["SQL_LENGTH"] = final_df["CUSTOM_SQL_TEXT"].astype(str).apply(len)
    final_df = final_df.sort_values(["RECORD_KEY","SQL_LENGTH"], ascending=[True, False])

    # duplicates snapshot
    dups_df = final_df[final_df.duplicated(subset=["RECORD_KEY"], keep=False)].copy()
    dups_df["DUP_GROUP"] = dups_df.groupby("RECORD_KEY").ngroup() + 1
    dups_df["DUP_COUNT"] = dups_df.groupby("RECORD_KEY")["RECORD_KEY"].transform("count")

    # keep first (longest SQL per key because of the sort above)
    final_df = final_df.drop_duplicates(subset="RECORD_KEY", keep="first").reset_index(drop=True)

    output_path = fr"{path}\SAP_DS_ALL_TABLE_MAPPING_{timestamp}.csv"
    dups_path   = fr"{path}\SAP_DS_TABLE_MAPPING_DUPLICATES_{timestamp}.csv"

    final_df.to_csv(output_path, index=False)
    dups_df.to_csv(dups_path, index=False)

    print(f"Done. Wrote: {output_path}  |  Rows: {len(final_df)}")
    print(f"Number of duplicate records: {len(dups_df)}")


if __name__ == "__main__":
    print("**** Process started at:", datetime.datetime.now())
    main()
    print("**** Process completed at:", datetime.datetime.now())
