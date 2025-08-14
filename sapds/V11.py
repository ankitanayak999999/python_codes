#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
import pandas as pd

# ------------------------ tiny utilities ------------------------

def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag) if isinstance(tag, str) else ""
def lower(s): return (s or "").strip().lower()
def attrs_ci(e): return {k.lower(): (v or "") for k, v in getattr(e, "attrib", {}).items()}

def build_parent_map(root): return {c: p for p in root.iter() for c in p}

def ancestors(e, pm, lim=200):
    cur = e
    for _ in range(lim):
        if cur is None: break
        yield cur
        cur = pm.get(cur)

def collect_text(n):
    parts=[]
    if hasattr(n,"attrib"): parts.extend([str(v) for v in n.attrib.values() if v])
    if n.text: parts.append(n.text)
    for c in list(n):
        parts.append(collect_text(c))
        if c.tail: parts.append(c.tail)
    return " ".join([p for p in parts if p])

def dedupe(seq):
    out, seen = [], set()
    for x in seq:
        if x is None: continue
        s = str(x).strip()
        if not s or s in seen: continue
        seen.add(s); out.append(s)
    return out

def canon(s: str) -> str: return re.sub(r'[^A-Z0-9]','',(s or '').upper())

# display normalization
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

# ------------------------ constants & detectors ------------------------

DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")

CACHE_WORDS  = {"PRE_LOAD_CACHE","POST_LOAD_CACHE","CACHE","PRE_LOAD","POST_LOAD"}
POLICY_WORDS = {"MAX","MIN","MAX-NS","MIN-NS","MAX_NS","MIN_NS"}
SPECIAL_TOKENS = CACHE_WORDS | POLICY_WORDS

NAME_CHARS    = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"
DOT_NORMALIZE = re.compile(r"\s*\.\s*")

# plain lookup detection (V8 behavior kept)
HAS_LOOKUP = re.compile(r'\blookup(?!_)\s*\(', re.I)

LOOKUP_CALL_RE     = re.compile(rf'\blookup(?!_)\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)
BRACED_TRIPLE      = re.compile(r'\blookup(?!_)\s*\(\s*\{\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^\}]+?)\s*\}', re.I|re.S)
LOOKUP_ARGS_RE     = re.compile(r'\blookup(?!_)\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*(?:,|\))', re.I|re.S)

# lookup_ext: FUNCTION_CALL and named kv only
LOOKUP_EXT_NAMED_KV_RE = re.compile(
    r'\blookup_ext\s*\([^)]*?'
    r'(?:tableDatastore|tabledatastore)\s*=\s*([\'"]?)(?P<ds>[^\'",)\s]+)\1[^)]*?'
    r'(?:tableOwner|tableowner)\s*=\s*([\'"]?)(?P<own>[^\'",)\s]+)\3[^)]*?'
    r'(?:tableName|tablename)\s*=\s*([\'"]?)(?P<tbl>[^\'",)\s]+)\5',
    re.I | re.S
)

def _is_token(t:str)->bool:
    return (t or "").strip().upper().replace(" ","_") in SPECIAL_TOKENS

def _looks_table(t:str)->bool:
    t = (t or "").strip().strip("'").strip('"')
    if not t or _is_token(t): return False
    if t.isdigit(): return False
    return bool(re.search(r"[A-Za-z]", t)) and len(t) >= 2

def _valid_triplet(ds, sch, tbl)->bool:
    return bool(ds) and _looks_table(tbl) and not _is_token(sch)

# -------- function name normalization / discovery --------

FUNC_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.I)

def normalize_fn_name(name: str) -> str:
    if not name: return ""
    s = str(name).strip()
    for sep in ("::", "/", "."):
        if sep in s:
            s = s.split(sep)[-1]
    return canon(s)

def extract_called_function_names(blob: str):
    names=set()
    if not blob: return names
    for m in FUNC_NAME_RE.finditer(blob):
        fn=(m.group(1) or "").strip()
        lo=fn.lower()
        if not fn or lo in ("lookup","lookup_ext"): continue
        names.add(fn)
    return names

# ------------------------ lookup extractors ------------------------

def extract_lookup_from_call(text: str, is_ext: bool = False):
    """Return (datastore, owner/schema, table) or ('','','')."""
    if not text: return ("","","")
    t = DOT_NORMALIZE.sub(".", text)

    if is_ext:
        mkv = LOOKUP_EXT_NAMED_KV_RE.search(t)
        if mkv and _valid_triplet(mkv.group("ds"), mkv.group("own"), mkv.group("tbl")):
            return mkv.group("ds").strip(), mkv.group("own").strip(), mkv.group("tbl").strip()

    m0 = (BRACED_TRIPLE if not is_ext else None)
    if m0:
        m = m0.search(t)
        if m and _valid_triplet(m.group(1), m.group(2), m.group(3)):
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    m2 = (LOOKUP_ARGS_RE if not is_ext else None)
    if m2:
        m = m2.search(t)
        if m and _valid_triplet(m.group(1), m.group(2), m.group(3)):
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    return ("","","")

def extract_all_lookup_ext_from_text(text: str):
    """All lookup_ext triplets from a function body using named kv."""
    out=[]
    if not text: return out
    for mkv in LOOKUP_EXT_NAMED_KV_RE.finditer(text):
        ds, sch, tb = mkv.group("ds"), mkv.group("own"), mkv.group("tbl")
        if _valid_triplet(ds, sch, tb):
            out.append((ds.strip(), sch.strip(), tb.strip()))
    return out

# ------------------------ project / job / df maps ------------------------

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
        if lower(strip_ns(getattr(p,"tag",""))) not in PROJECT_TAGS: continue
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

# ------------------------ schema/column helpers ------------------------

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

# ------------------------ SQL helpers ------------------------

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

# ------------------------ main parser (V8 base + focused fixes) ------------------------

Record = namedtuple(
    "Record",
    [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table",
        "transformation_position","transformation_usage_count","custom_sql_text",
    ],
)

def parse_single_xml(xml_path: str) -> pd.DataFrame:
    tree=ET.parse(xml_path); root=tree.getroot()
    pm=build_parent_map(root)

    # multi-project helpers
    job_to_project = build_job_to_project_map(root)
    df_to_project  = build_df_project_map(root)

    # function bodies (canonical-keyed)
    function_bodies = {}
    FUNCTION_DEF_TAGS = {
        "dicustomfunction","difunction","function","diprocedure",
        "userfunction","diuserfunction","diuserdefinedfunction","scriptfunction","discriptfunction"
    }
    for node in root.iter():
        tag  = lower(strip_ns(getattr(node,"tag","")))
        if tag not in FUNCTION_DEF_TAGS: continue
        name = (getattr(node,"attrib",{}).get("name") or
                getattr(node,"attrib",{}).get("displayName") or "").strip()
        if not name: 
            continue
        txt = collect_text(node)
        function_bodies[ normalize_fn_name(name) ] = txt

    # display caches
    display_ds  = defaultdict(NameBag)
    display_sch = defaultdict(NameBag)
    display_tbl = defaultdict(NameBag)
    def remember_display(ds, sch, tbl):
        ds=_strip_wrappers(ds).strip()
        sch=_strip_wrappers(sch).strip()
        tbl=_strip_wrappers(tbl).strip()
        k=(_norm_key(ds), _norm_key(sch), _norm_key(tbl))
        display_ds[k].add(ds); display_sch[k].add(sch); display_tbl[k].add(tbl)

    # collectors
    lookup_pos     = defaultdict(list)  # (proj,job,df,ds,sch,tbl) -> ["Schema>>Col", ...]
    lookup_ext_pos = defaultdict(set)   # (proj,job,df,ds,sch,tbl) -> {"Schema", ...}
    source_target  = set()
    sql_rows       = []                 # custom SQL as synthetic rows

    # external function attribution
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

    # walk xml
    for e in root.iter():
        if not isinstance(e.tag, str)): 
            continue
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

        # sources / targets (DB)
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

        # sources / targets (FILE)
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

        # -------- custom SQL capture (V8 behavior) --------
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
                disp_name=""
                for up in ancestors(e, pm, 12):
                    at=attrs_ci(up); tt=lower(strip_ns(getattr(up,"tag","")))
                    if tt=="diattribute" and lower(at.get("name","")) in ("ui_display_name","ui_acta_from_schema_0"):
                        disp_name=at.get("value","").strip() or disp_name
                    if tt=="dischema" and not disp_name:
                        disp_name=(at.get("name") or "").strip() or disp_name

                tables=extract_tables_from_sql(sql_text)
                table_csv=", ".join(tables) if tables else "SQL_TEXT"
                ds_for_sql=""
                for up in ancestors(e, pm, 12):
                    for ch in up.iter():
                        if lower(strip_ns(getattr(ch,"tag","")))=="diattribute" and lower(ch.attrib.get("name",""))=="database_datastore":
                            ds_for_sql=(ch.attrib.get("value") or "").strip()
                            if ds_for_sql: break
                    if ds_for_sql: break
                ds_for_sql = ds_for_sql or "DS_SQL"

                remember_display(ds_for_sql,"CUSTOM_SQL",table_csv)
                sql_rows.append((
                    proj,job,df,"custom_sql",
                    ds_for_sql,"CUSTOM_SQL",table_csv,
                    disp_name or "SQL",
                    len(tables),
                    '"' + sql_text.replace('"','""') + '"',
                ))

        # ------- lookup (column-level)  [V8 logic kept] -------
        if tag=="diattribute" and lower(a.get("name",""))=="ui_mapping_text":
            txt=a.get("value") or e.text or ""
            if HAS_LOOKUP.search(txt):
                proj,job,df=context_for(e)
                schema_out=schema_out_from_DISchema(e, pm, cur_schema)
                col=find_output_column(e, pm)
                dsl,schl,tbl=extract_lookup_from_call(txt, is_ext=False)
                if dsl and tbl and schema_out and col:
                    remember_display(dsl,schl,tbl)
                    key=(proj,job,df,_norm_key(dsl),_norm_key(schl),_norm_key(tbl))
                    lookup_pos[key].append(f"{schema_out}>>{col}")
                elif schema_out and col:
                    # lookup present but we couldn't parse ds/sch/tbl
                    key=(proj,job,df,_norm_key(""),_norm_key(""),_norm_key(""))
                    lookup_pos[key].append(f"{schema_out}>>{col}")

        # ------- lookup_ext (FUNCTION_CALL authoritative ONLY) -------
        if tag=="function_call" and lower(a.get("name",""))=="lookup_ext":
            proj,job,df=context_for(e)
            schema_out=schema_out_from_DISchema(e, pm, cur_schema)
            # two ways to get triplet: attributes (named kv) or inner text
            txt = " ".join([f'{k}="{v}"' for k,v in a.items()]) + " " + collect_text(e)
            dsx=schx=tbx=""
            m=LOOKUP_EXT_NAMED_KV_RE.search(txt)
            if m and _valid_triplet(m.group("ds"),m.group("own"),m.group("tbl")):
                dsx,schx,tbx = m.group("ds"), m.group("own"), m.group("tbl")
            if not (dsx and tbx):
                # could not parse triplet; still record presence
                pos = ""
                col = find_output_column(e, pm)
                if schema_out and col:
                    pos = f"{schema_out}>>{col}"
                elif schema_out:
                    pos = schema_out
                key=(proj,job,df,_norm_key(""),_norm_key(""),_norm_key(""))
                if pos:
                    lookup_ext_pos[key].add(pos)
                continue

            remember_display(dsx,schx,tbx)
            key=(proj,job,df,_norm_key(dsx),_norm_key(schx),_norm_key(tbx))
            # transformation position: prefer Schema>>Column when inside DIElement; else Schema name
            col=find_output_column(e, pm)
            pos = f"{schema_out}>>{col}" if (schema_out and col) else (schema_out or "")
            if pos:
                lookup_ext_pos[key].add(pos)

        # ------- record calls to external functions so we can attribute their internal lookups -------
        if tag in ("diexpression","diattribute","function_call"):
            blob = " ".join([f'{k}="{v}"' for k, v in a.items()]) + " " + collect_text(e)
            proj,job,df=context_for(e)
            schema_out=schema_out_from_DISchema(e, pm, cur_schema)
            col=find_output_column(e, pm)
            if schema_out:
                for fn in extract_called_function_names(blob):
                    pos=f"{schema_out}>>{col}" if col else schema_out
                    df_func_positions[df][ normalize_fn_name(fn) ].add(pos)

    # ---- expand external function lookups into DF collectors (normalized names) ----
    for df_name, fn_map in df_func_positions.items():
        proj,job = df_context.get(df_name, ("",""))
        if not proj:
            proj = job_to_project.get(job, "") or df_to_project.get(df_name, "")
        for fn_key, positions in fn_map.items():
            body = function_bodies.get(fn_key, "")
            if not body: continue
            # lookup_ext inside the function body -> add for each calling position (schema-level)
            for ds,sch,tb in extract_all_lookup_ext_from_text(body):
                remember_display(ds,sch,tb)
                key=(proj,job,df_name,_norm_key(ds),_norm_key(sch),_norm_key(tb))
                for p in {pos.split(">>",1)[0] for pos in positions if pos}:  # schema-only for ext
                    lookup_ext_pos[key].add(p)
            # plain lookup inside the function body -> column-level
            if HAS_LOOKUP.search(body):
                dsl,schl,tbl=extract_lookup_from_call(body, is_ext=False)
                if dsl and tbl:
                    remember_display(dsl,schl,tbl)
                    key=(proj,job,df_name,_norm_key(dsl),_norm_key(schl),_norm_key(tbl))
                    for p in positions:
                        lookup_pos[key].append(p)
                else:
                    # lookup present but could not parse triplet: still mark
                    key=(proj,job,df_name,_norm_key(""),_norm_key(""),_norm_key(""))
                    for p in positions:
                        lookup_pos[key].append(p)

    # ---------- finalize rows ----------
    def nice_names(dsN, schN, tblN):
        k=(dsN, schN, tblN)
        return ( display_ds[k].get(dsN), display_sch[k].get(schN), display_tbl[k].get(tblN) )

    Row = namedtuple(
        "Row",
        [
            "project_name","job_name","dataflow_name","role",
            "datastore","schema","table",
            "transformation_position","transformation_usage_count","custom_sql_text",
        ],
    )

    rows=[]

    # lookups
    for (proj,job,df,dsN,schN,tblN), positions in lookup_pos.items():
        uniq=sorted(dedupe(positions))
        if not uniq: continue
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        rows.append(Row(proj or "", job or "", df or "", "lookup",
                        dsD, schD, tblD, ", ".join(uniq), len(uniq), ""))

    # lookup_ext
    for (proj,job,df,dsN,schN,tblN), posset in lookup_ext_pos.items():
        uniq=sorted(dedupe(list(posset)))
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        rows.append(Row(proj or "", job or "", df or "", "lookup_ext",
                        dsD, schD, tblD, ", ".join(uniq), len(uniq), ""))

    # sources / targets
    for (proj,job,df,role,dsN,schN,tblN) in sorted(set([]).union(source_target)):
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        rows.append(Row(proj or "", job or "", df or "", role,
                        dsD, schD, tblD, "", 0, ""))

    # custom SQL rows
    for r in sql_rows:
        rows.append(Row(*r))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # backfill project via job map, then via DF containment
    def fill_proj(row):
        if str(row["project_name"]).strip():
            return row["project_name"]
        j = row.get("job_name","")
        d = row.get("dataflow_name","")
        if j and j in job_to_project:
            return job_to_project[j]
        return df_to_project.get(d, "")

    df["project_name"] = df.apply(fill_proj, axis=1)

    # strict de-dupe & merge positions
    def nkey(r):
        return (
            r["project_name"], r["job_name"], r["dataflow_name"], r["role"],
            _norm_key(r["datastore"]), _norm_key(r["schema"]), _norm_key(r["table"]),
            _norm_key(r.get("custom_sql_text",""))
        )

    df["__k__"]=df.apply(nkey, axis=1)

    def merge_pos(series):
        return ", ".join(sorted(dedupe([p.strip() for p in series if str(p).strip()])))

    df=(df.groupby(
            ["__k__","project_name","job_name","dataflow_name","role",
             "datastore","schema","table","custom_sql_text"],
            dropna=False, as_index=False)
         .agg({"transformation_position": merge_pos,
               "transformation_usage_count": "sum"}))
    df=df.drop(columns=["__k__"])

    # cleanup & sort
    for c in ("datastore","schema","table","custom_sql_text","transformation_position"):
        df[c]=df[c].map(_pretty)

    df=df.sort_values(by=["project_name","job_name","dataflow_name","role","datastore","schema","table"]).reset_index(drop=True)
    return df

# ------------------------ main ------------------------

def main():
    # set these paths (keep your current defaults if you want)
    xml_path = r"C:\Users\raksahu\Downloads\python\input\export_afs.xml"
    out_xlsx = r"C:\Users\raksahu\Downloads\python\input\output_v9_afs.xlsx"

    df = parse_single_xml(xml_path)

    cols = [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table",
        "transformation_position","transformation_usage_count","custom_sql_text",
    ]
    for c in cols:
        if c not in df.columns: df[c]=""
    df = df[cols]

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print(f"Done. Wrote: {out_xlsx}  |  Rows: {len(df)}")

if __name__ == "__main__":
    main()
