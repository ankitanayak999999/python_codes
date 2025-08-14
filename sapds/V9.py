#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from collections import defaultdict, namedtuple
import pandas as pd
from lxml import etree as ET

# ================================================================
# Utilities
# ================================================================

def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag) if isinstance(tag, str) else ""
def lower(s): return (s or "").strip().lower()
def attrs_ci(e): return {k.lower(): (v or "") for k, v in (getattr(e, "attrib", {}) or {}).items()}
def line_no(node): return getattr(node, "sourceline", -1) or -1

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

# ================================================================
# Constants
# ================================================================

DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
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

# ================================================================
# Project / Job / DF mapping helpers
# ================================================================

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
        for dn in df_names: df_proj.setdefault(dn, only)
    return df_proj

# ================================================================
# DISchema / Element helpers (for transformation positions)
# ================================================================

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
    # Direct element?
    if lower(strip_ns(getattr(e,"tag","")))=="dielement":
        nm=(attrs_ci(e).get("name") or "").strip()
        if nm: return nm
    # Walk up to find the enclosing DIElement
    cur=e
    for _ in range(200):
        if cur is None: break
        if lower(strip_ns(cur.tag))=="dielement":
            nm=(attrs_ci(cur).get("name") or "").strip()
            if nm: return nm
        cur=pm.get(cur)
    return ""

# ================================================================
# SQL helpers
# ================================================================

SQL_FROM_JOIN_RE = re.compile(r"\b(?:from|join)\s+([A-Za-z0-9_\.\$#@]+)", re.I)

def extract_tables_from_sql(sql_text: str):
    if not sql_text: return []
    c = " ".join(sql_text.replace("\n"," ").replace("\r"," ").split())
    hits = SQL_FROM_JOIN_RE.findall(c)
    tables=[]
    for h in hits:
        parts=h.split(".")
        if len(parts)>=2:
            tables.append(f"{parts[-2]}.{parts[-1]}")
        else:
            tables.append(parts[-1])
    return dedupe(tables)

# ================================================================
# Main parser
# ================================================================

def parse_single_xml(xml_path: str) -> pd.DataFrame:
    parser = ET.XMLParser(huge_tree=True, recover=True)
    tree   = ET.parse(xml_path, parser=parser)
    root   = tree.getroot()
    pm     = build_parent_map(root)

    job_to_project = build_job_to_project_map(root)
    df_to_project  = build_df_project_map(root)

    # friendly-name caches
    display_ds  = defaultdict(NameBag)
    display_sch = defaultdict(NameBag)
    display_tbl = defaultdict(NameBag)
    def remember_display(ds, sch, tbl):
        ds=_strip_wrappers(ds).strip()
        sch=_strip_wrappers(sch).strip()
        tbl=_strip_wrappers(tbl).strip()
        k=(_norm_key(ds), _norm_key(sch), _norm_key(tbl))
        display_ds[k].add(ds); display_sch[k].add(sch); display_tbl[k].add(tbl)

    # collectors ---------------------------------------------------
    source_target  = set()                            # (proj, job, df, role, dsN, schN, tblN)
    sql_rows       = []                               # list of Record tuples
    lookup_pos     = defaultdict(lambda: defaultdict(set))   # key -> pos -> set(lines)
    lookup_ext_pos = defaultdict(lambda: defaultdict(set))   # key -> pos -> set(lines)
    missing_lookup = []                               # rows with missing DS/Owner/Table

    df_context        = {}  # df -> (proj,job)
    df_func_positions = defaultdict(lambda: defaultdict(set))  # df -> canon(fn) -> {positions}

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
        job = job or (last_job if not cur_job else cur_job)
        proj = job_to_project.get(job, proj or df_to_project.get(df, cur_proj))
        if df and (df not in df_context): df_context[df]=(proj or "", job or "")
        return proj or "", job or "", df or ""

    # -------- scan function definitions for their own FUNCTION_CALL lookups
    def normalize_fn_name(name: str) -> str:
        if not name: return ""
        s = name.strip()
        for sep in ("::","/","."):
            if sep in s: s=s.split(sep)[-1]
        return re.sub(r'[^A-Z0-9]','',s.upper())

    fn_lookup_calls = defaultdict(list)   # canon(func) -> list of (role, ds, sch, tbl, ln)
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
                ln   = line_no(fc)
                fn_lookup_calls[key].append((role, ds, sch, tbl, ln))

    # -------- walk XML once (organized blocks)

    for e in root.iter():
        if not isinstance(e.tag, str): continue
        tag=lower(strip_ns(e.tag)); a=attrs_ci(e)

        # ---- context tracking
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

        # ---- 1) SOURCES / TARGETS (DB)
        if tag in ("didatabasetablesource","didatabasetabletarget"):
            ds =(a.get("datastorename") or a.get("datastore") or "").strip()
            sch=(a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl=(a.get("tablename") or a.get("table") or "").strip()
            if ds and tbl:
                remember_display(ds,sch,tbl)
                proj,job,df=context_for(e)
                role="source" if "source" in tag else "target"
                key=(proj,job,df,role,_norm_key(ds),_norm_key(sch),_norm_key(tbl))
                source_target.add(key)

        # ---- 2) SOURCES / TARGETS (FILE)
        if tag in ("difilesource","difiletarget"):
            proj,job,df=context_for(e)
            role="source" if "source" in tag else "target"
            fmt=(a.get("formatname") or "").strip()
            fname=(a.get("filename") or a.get("name") or "").strip()
            ds=(a.get("datastorename") or a.get("datastore") or "FILE").strip() or "FILE"
            sch=fmt or "FILE"
            tbl=fname or "FILE_OBJECT"
            remember_display(ds,sch,tbl)
            key=(proj,job,df,role,_norm_key(ds),_norm_key(sch),_norm_key(tbl))
            source_target.add(key)

        # ---- 3) CUSTOM SQL (kept same behavior as V8)
        if tag in ("sqltext","sqltexts","diquery","ditransformcall"):
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
                # position name
                disp_name=""
                for up in ancestors(e, pm, 12):
                    at=attrs_ci(up); tt=lower(strip_ns(getattr(up,"tag","")))
                    if tt=="diattribute" and lower(at.get("name","")) in ("ui_display_name","ui_acta_from_schema_0"):
                        disp_name=at.get("value","").strip() or disp_name
                    if tt=="dischema" and not disp_name:
                        disp_name=(at.get("name") or "").strip() or disp_name
                # simple table list from SQL
                tables=extract_tables_from_sql(sql_text)
                table_csv=", ".join(tables) if tables else "SQL_TEXT"
                # datastore near SQL (heuristic)
                ds_for_sql=""
                for up in ancestors(e, pm, 12):
                    for ch in up.iter():
                        if lower(strip_ns(getattr(ch,"tag","")))=="diattribute" and lower(ch.attrib.get("name",""))=="database_datastore":
                            ds_for_sql=(ch.attrib.get("value") or "").strip()
                            if ds_for_sql: break
                    if ds_for_sql: break
                ds_for_sql = ds_for_sql or "DS_SQL"
                remember_display(ds_for_sql,"CUSTOM_SQL",table_csv)
                sql_rows.append(Record(
                    proj,job,df,"custom_sql",
                    ds_for_sql,"CUSTOM_SQL",table_csv,
                    disp_name or "SQL", len(tables),
                    '"' + (sql_text.replace('"','""')) + '"',
                    line_no(e),
                ))

        # ---- 4) LOOKUPS FROM <FUNCTION_CALL> ONLY
        if tag=="function_call":
            proj,job,df = context_for(e)
            an  = attrs_ci(e)
            cal = (an.get("name") or "").strip().lower()
            if cal not in ("lookup","lookup_ext"):
                # record external function positions for later expansion
                # (this covers functions that themselves contain lookups)
                schema_out = schema_out_from_DISchema(e, pm, cur_schema)
                col        = find_output_column(e, pm)
                if schema_out:
                    pos = f"{schema_out}>>{col}" if col else schema_out
                    fn_key = normalize_fn_name(an.get("name") or "")
                    df_func_positions[df][fn_key].add(pos)
                continue

            # Position for this specific lookup call
            schema_out = schema_out_from_DISchema(e, pm, cur_schema)
            col        = find_output_column(e, pm)
            pos        = f"{schema_out}>>{col}" if (schema_out and col) else (schema_out or "")

            # Pull DS/Owner/Table only from FUNCTION_CALL attributes
            ds  = an.get("tabledatastore") or ""
            sch = an.get("tableowner") or ""
            tbl = an.get("tablename") or ""
            ln  = line_no(e)

            # If missing DS/schema/table, still emit a placeholder row
            if not (ds and tbl):
                missing_lookup.append(Record(
                    proj, job, df, "lookup_ext" if cal=="lookup_ext" else "lookup",
                    ds or "<missing>", sch or "<missing>", tbl or "<missing>",
                    pos, 1, "", ln
                ))
            else:
                remember_display(ds, sch, tbl)
                key = (proj, job, df, _norm_key(ds), _norm_key(sch), _norm_key(tbl))
                if cal == "lookup_ext":
                    lookup_ext_pos[key][pos].add(ln)
                else:
                    lookup_pos[key][pos].add(ln)

            # Also note that this DF contains a callsite of a function by this name,
            # so if this particular call was to a *user function* we will expand later.
            if cal not in ("lookup","lookup_ext"):
                fn_key = normalize_fn_name(an.get("name") or "")
                if fn_key:
                    schema_for_fn = schema_out_from_DISchema(e, pm, cur_schema)
                    col_for_fn    = find_output_column(e, pm)
                    pos2          = f"{schema_for_fn}>>{col_for_fn}" if (schema_for_fn and col_for_fn) else (schema_for_fn or "")
                    if pos2:
                        df_func_positions[df][fn_key].add(pos2)

    # ---- 5) Expand lookups that live inside external function definitions
    for df_name, fn_map in df_func_positions.items():
        proj,job = df_context.get(df_name, ("",""))
        if not proj:
            proj = job_to_project.get(job, "") or df_to_project.get(df_name, "")
        for fn_key, positions in fn_map.items():
            for role, ds, sch, tbl, ln in fn_lookup_calls.get(fn_key, []):
                dsN, schN, tblN = _norm_key(ds), _norm_key(sch), _norm_key(tbl)
                if not ds or not tbl:
                    # even if missing, emit placeholders so you can see the event
                    for p in positions:
                        missing_lookup.append(Record(
                            proj, job, df_name, role,
                            ds or "<missing>", sch or "<missing>", tbl or "<missing>",
                            p, 1, "", ln
                        ))
                    continue
                remember_display(ds, sch, tbl)
                key = (proj, job, df_name, dsN, schN, tblN)
                if role == "lookup_ext":
                    for p in positions:
                        lookup_ext_pos[key][p].add(ln)
                else:
                    for p in positions:
                        lookup_pos[key][p].add(ln)

    # ================================================================
    # Finalize rows (merge positions, dedupe, sort)
    # ================================================================

    def nice_names(dsN, schN, tblN):
        k=(dsN, schN, tblN)
        return (display_ds[k].get(dsN), display_sch[k].get(schN), display_tbl[k].get(tblN))

    rows=[]

    # lookups (column-level)
    for (proj,job,df,dsN,schN,tblN), posmap in lookup_pos.items():
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        positions=[]
        lines=[]
        for p, lnset in posmap.items():
            if p: positions.append(p)
            lines.extend(sorted(lnset))
        positions = sorted(dedupe(positions))
        rows.append(Record(proj, job, df, "lookup", dsD, schD, tblD,
                           ", ".join(positions), len(positions), "", min(lines) if lines else -1))

    # lookup_ext (schema-level)
    for (proj,job,df,dsN,schN,tblN), posmap in lookup_ext_pos.items():
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        positions=[]
        lines=[]
        for p, lnset in posmap.items():
            if p: positions.append(p)
            lines.extend(sorted(lnset))
        positions = sorted(dedupe(positions))
        rows.append(Record(proj, job, df, "lookup_ext", dsD, schD, tblD,
                           ", ".join(positions), len(positions), "", min(lines) if lines else -1))

    # missing-info lookups (placeholders)
    rows.extend(missing_lookup)

    # sources / targets
    for (proj,job,df,role,dsN,schN,tblN) in sorted(source_target):
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        rows.append(Record(proj, job, df, role, dsD, schD, tblD, "", 0, "", -1))

    # custom SQL rows
    rows.extend(sql_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # fill project if blank (job map, then DF containment)
    def fill_proj(row):
        if str(row["project_name"]).strip(): return row["project_name"]
        j = row.get("job_name",""); d = row.get("dataflow_name","")
        if j and j in job_to_project: return job_to_project[j]
        return df_to_project.get(d, "")

    df["project_name"] = df.apply(fill_proj, axis=1)

    # strict de-dup: same keys + same position should merge usage_count and keep min line
    def nkey(r):
        return (
            r["project_name"], r["job_name"], r["dataflow_name"], r["role"],
            _norm_key(r["datastore"]), _norm_key(r["schema"]), _norm_key(r["table"]),
            _norm_key(r.get("custom_sql_text","")), _norm_key(r.get("transformation_position",""))
        )
    df["__k__"]=df.apply(nkey, axis=1)

    def merge_pos(series):  # we already key by exact position, so this is trivial; still normalize
        return ", ".join(sorted(dedupe([p.strip() for p in series if str(p).strip()])))

    df=(df.groupby(
            ["__k__","project_name","job_name","dataflow_name","role",
             "datastore","schema","table","custom_sql_text","transformation_position"],
            dropna=False, as_index=False)
         .agg({"transformation_usage_count": "sum",
               "source_line": "min"}))

    df=df.drop(columns=["__k__"])

    # cleanup & sort
    for c in ("datastore","schema","table","custom_sql_text","transformation_position"):
        df[c]=df[c].map(_pretty)

    df=df.sort_values(by=[
        "project_name","job_name","dataflow_name","role","datastore","schema","table","transformation_position"
    ]).reset_index(drop=True)
    return df

# ================================================================
# Main
# ================================================================

def main():
    # keep these defaults (you can change them before running)
    xml_path = r"C:\Users\raksahu\Downloads\python\input\export_afs.xml"
    out_xlsx = r"C:\Users\raksahu\Downloads\python\input\output_v9_afs.xlsx"

    df = parse_single_xml(xml_path)

    cols = [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table",
        "transformation_position","transformation_usage_count",
        "custom_sql_text","source_line",
    ]
    for c in cols:
        if c not in df.columns: df[c] = ""
    df = df[cols]

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print(f"Done. Wrote: {out_xlsx}  |  Rows: {len(df)}")

if __name__ == "__main__":
    main()
