#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
import pandas as pd
from typing import Dict, Tuple, Iterable

# ------------------------ tiny utilities ------------------------

def strip_ns(tag): 
    return re.sub(r"^\{.*\}", "", tag) if isinstance(tag, str) else ""

def lower(s): 
    return (s or "").strip().lower()

def attrs_ci(e) -> Dict[str, str]:
    return { (k or "").lower(): (v or "") for k, v in getattr(e, "attrib", {}).items() }

def build_parent_map(root): 
    return {c: p for p in root.iter() for c in p}

def ancestors(e, pm, lim=200) -> Iterable:
    cur = e
    for _ in range(lim):
        if cur is None: break
        yield cur
        cur = pm.get(cur)

def collect_text(n):
    parts=[]
    if hasattr(n,"attrib"):
        parts.extend([str(v) for v in n.attrib.values() if v])
    if n.text: 
        parts.append(n.text)
    for c in list(n):
        parts.append(collect_text(c))
        if c.tail: parts.append(c.tail)
    return " ".join([p for p in parts if p])

def dedupe(seq):
    out, seen = [], set()
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def canon(s:str)->str: 
    return re.sub(r'[^A-Z0-9]','',(s or '').upper())

# ------------------------ display normalization ------------------------

def _strip_wrappers(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().strip('"').strip("'")
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        s = s[1:-1]
    return s

def _norm_key(s: str) -> str:
    s = _strip_wrappers(s)
    s = re.sub(r"[\{\}\[\]]", "", s)
    s = re.sub(r"\s+", " ", s).replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s.upper()

def _pretty(s: str) -> str:
    s = _strip_wrappers(s)
    s = re.sub(r"[\{\}\[\]]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
CALLSTEP_TAGS = ("dicallstep","callstep")

# classification helpers
HAS_LOOKUP     = re.compile(r'\blookup(?!_)\s*\(', re.I)

# lookup (standard) extractors
NAME_CHARS    = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"
DOT_NORMALIZE = re.compile(r"\s*\.\s*")
LOOKUP_CALL_RE     = re.compile(rf'\blookup(?!_)\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)
BRACED_TRIPLE      = re.compile(r'\blookup(?!_)\s*\(\s*\{\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^\}]+?)\s*\}', re.I|re.S)
LOOKUP_ARGS_RE     = re.compile(r'\blookup(?!_)\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*(?:,|\))', re.I|re.S)

# lookup_ext from FUNCTION_CALL (authoritative) and flexible parsing ONLY when tag is FUNCTION_CALL
LOOKUP_EXT_NAMED_KV_RE = re.compile(
    r'\blookup_ext\s*\([^)]*?'
    r'(?:tableDatastore|tabledatastore)\s*=\s*([\'"]?)(?P<ds>[^\'",)\s]+)\1[^)]*?'
    r'(?:tableOwner|tableowner)\s*=\s*([\'"]?)(?P<own>[^\'",)\s]+)\3[^)]*?'
    r'(?:tableName|tablename)\s*=\s*([\'"]?)(?P<tbl>[^\'",)\s]+)\5',
    re.I | re.S
)
LOOKUP_EXT_ARGS_RE = re.compile(r'\blookup_ext\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*(?:,|\))', re.I|re.S)
LOOKUP_EXT_DOTTED_RE = re.compile(rf'\blookup_ext\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)

CACHE_WORDS  = {"PRE_LOAD_CACHE","POST_LOAD_CACHE","CACHE","PRE_LOAD","POST_LOAD"}
POLICY_WORDS = {"MAX","MIN","MAX-NS","MIN-NS","MAX_NS","MIN_NS"}

def _is_cache_or_policy(token: str) -> bool:
    t = (token or "").strip().strip("'").strip('"').upper().replace(" ","_")
    return t in CACHE_WORDS or t in POLICY_WORDS

def _looks_like_table(token: str) -> bool:
    t = (token or "").strip().strip("'").strip('"')
    if not t or _is_cache_or_policy(t): return False
    if t.isdigit(): return False
    return bool(re.search(r"[A-Za-z]", t)) and len(t) >= 3

def _valid_triplet(ds: str, sch: str, tbl: str) -> bool:
    return bool(ds) and _looks_like_table(tbl) and not _is_cache_or_policy(sch)

def extract_lookup_from_text(text: str):
    """Return (datastore, owner/schema, table) for standard lookup."""
    if not text: return ("","","")
    t = DOT_NORMALIZE.sub(".", text)

    m0 = BRACED_TRIPLE.search(t)
    if m0 and _valid_triplet(m0.group(1), m0.group(2), m0.group(3)):
        return m0.group(1).strip(), m0.group(2).strip(), m0.group(3).strip()

    m1 = LOOKUP_CALL_RE.search(t)
    if m1 and _valid_triplet(m1.group(1), m1.group(2), m1.group(3)):
        return m1.group(1).strip(), m1.group(2).strip(), m1.group(3).strip()

    m2 = LOOKUP_ARGS_RE.search(t)
    if m2 and _valid_triplet(m2.group(1), m2.group(2), m2.group(3)):
        return m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()

    return ("","","")

def extract_lookup_ext_from_function_call(node_text: str, attrs: Dict[str,str]):
    """Return (datastore, owner/schema, table) only from FUNCTION_CALL (no DIExpression fallback)."""
    # 1) try named kv in attributes
    a_str = " ".join([f'{k}="{v}"' for k,v in attrs.items()])
    m = LOOKUP_EXT_NAMED_KV_RE.search(a_str)
    if m and _valid_triplet(m.group("ds"), m.group("own"), m.group("tbl")):
        return m.group("ds").strip(), m.group("own").strip(), m.group("tbl").strip()

    # 2) dotted or args inside collected node text
    t = DOT_NORMALIZE.sub(".", node_text or "")
    m1 = LOOKUP_EXT_DOTTED_RE.search(t)
    if m1 and _valid_triplet(m1.group(1), m1.group(2), m1.group(3)):
        return m1.group(1).strip(), m1.group(2).strip(), m1.group(3).strip()

    m2 = LOOKUP_EXT_ARGS_RE.search(t)
    if m2 and _valid_triplet(m2.group(1), m2.group(2), m2.group(3)):
        return m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()

    return ("","","")

# ------------------------ names (project / job / df) ------------------------

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

def build_df_project_map(root):
    df_names = collect_df_names(root)
    projects=[]
    for p in root.iter():
        if lower(strip_ns(getattr(p,"tag",""))) in PROJECT_TAGS:
            nm = (p.attrib.get("name") or p.attrib.get("displayName") or "").strip()
            if nm: projects.append((nm,p))
    df_proj={}

    # project contains DIJobRef -> jobs -> callsteps -> dataflows is complex; many exports simply nest DFs under project
    # so: record any DF nested under each project
    for nm, p in projects:
        for d in p.iter():
            if lower(strip_ns(getattr(d,"tag",""))) in DF_TAGS:
                dn=(d.attrib.get("name") or d.attrib.get("displayName") or "").strip()
                if dn: df_proj.setdefault(dn, nm)

    if len(projects)==1:
        only=projects[0][0]
        for dn in df_names: df_proj.setdefault(dn, only)

    return df_proj

def build_df_job_map(root):
    df_names = collect_df_names(root); df_canon={canon(n):n for n in df_names}
    jobs={}; wfs={}
    for n in root.iter():
        t=lower(strip_ns(getattr(n,"tag","")))
        if t in JOB_TAGS:
            nm=_job_name_from_node(n)
            if nm: jobs[nm]=n
        elif t in WF_TAGS:
            nm=(n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: wfs[nm]=n

    pm=build_parent_map(root)
    edges=defaultdict(set)

    def add_edge(src_kind, src_name, dst_kind, dst_name):
        if src_name and dst_name: edges[(src_kind, canon(src_name))].add((dst_kind, canon(dst_name)))

    for cs in root.iter():
        if lower(strip_ns(getattr(cs,"tag",""))) not in CALLSTEP_TAGS: continue

        # find source (job/workflow) up the tree
        src_kind, src_name = None, None
        cur=cs
        for _ in range(200):
            cur=pm.get(cur) 
            if not cur: break
            t=lower(strip_ns(cur.tag))
            if t in JOB_TAGS: src_kind,src_name="job",_job_name_from_node(cur); break
            if t in WF_TAGS:  src_kind,src_name="wf",(cur.attrib.get("name") or cur.attrib.get("displayName") or ""); break
        if not src_name: continue

        a=attrs_ci(cs); vals=list(a.values()); vals.append(collect_text(cs))
        tgt_type=(a.get("calledobjecttype") or a.get("type") or "").strip().lower()
        names=[]
        for k in ("calledobject","name","object","target","called_object"):
            if a.get(k):
                raw=a.get(k); names.append(raw)
                if any(sep in raw for sep in ["/","\\",".",":"]):
                    names.append(raw.split("/")[-1].split("\\")[-1].split(":")[-1].split(".")[-1])

        txt=" ".join(vals)

        if tgt_type in ("workflow","diworkflow"):
            for nm in names: add_edge(src_kind,src_name,"wf",nm)
        elif tgt_type in ("dataflow","didataflow"):
            for nm in names: add_edge(src_kind,src_name,"df",nm)
        else:
            can=canon(txt)
            for w in wfs.keys():
                if canon(w) in can: add_edge(src_kind,src_name,"wf",w)
            for d in df_names:
                if canon(d) in can: add_edge(src_kind,src_name,"df",d)

    df_job={}
    for j in jobs.keys():
        start=("job", canon(j)); seen={start}; stack=[start]; reach=set()
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

# ------------------------ main parser ------------------------

Record = namedtuple("Record", [
    "project_name","job_name","dataflow_name",
    "role","datastore","schema","table",
    "transformation_position","transformation_usages_count","custom_sql_text"
])

SQL_TABLE_RE = re.compile(r'\b(?:from|join)\s+([A-Z0-9_\."]+)', re.I)

def parse_single_xml(xml_path: str) -> pd.DataFrame:
    tree=ET.parse(xml_path); root=tree.getroot()
    pm=build_parent_map(root)

    df_job_map  = build_df_job_map(root)
    df_proj_map = build_df_project_map(root)

    # pretty name caches
    display_ds  = defaultdict(NameBag)
    display_sch = defaultdict(NameBag)
    display_tbl = defaultdict(NameBag)

    def remember_display(ds, sch, tbl):
        ds = _strip_wrappers(ds).strip()
        sch= _strip_wrappers(sch).strip()
        tbl= _strip_wrappers(tbl).strip()
        k=(_norm_key(ds),_norm_key(sch),_norm_key(tbl))
        display_ds[k].add(ds); display_sch[k].add(sch); display_tbl[k].add(tbl)

    source_target=set()
    lookup_pos    = defaultdict(list)  # schema>>column (standard lookup)
    lookup_ext_pos= defaultdict(set)   # schema (ext only)
    sql_rows      = []                 # custom SQL capture

    cur_proj=cur_job=cur_df=cur_schema=""; last_job=""

    def context_for(e):
        proj=job=df=""
        for a in ancestors(e, pm):
            t=lower(strip_ns(a.tag)); at=attrs_ci(a)
            nm=(at.get("name") or at.get("displayname") or "").strip()
            if not df and t in DF_TAGS: df=nm or df
            if not proj and t in PROJECT_TAGS: proj=nm or proj
            if t in JOB_TAGS and not job: job=_job_name_from_node(a) or job
        df=df or cur_df
        proj=proj or df_proj_map.get(df, cur_proj)
        job=job or df_job_map.get(df, cur_job or last_job)
        return proj, job, df

    for e in root.iter():
        if not isinstance(e.tag, str):
            continue
        tag=lower(strip_ns(e.tag)); a=attrs_ci(e)

        if tag in PROJECT_TAGS: cur_proj=(a.get("name") or a.get("displayname") or cur_proj).strip()
        if tag in DF_TAGS:      cur_df  =(a.get("name") or a.get("displayname") or cur_df).strip()
        if tag in JOB_TAGS:
            cur_job=_job_name_from_node(e) or (a.get("name") or a.get("displayname") or cur_job).strip()
            if cur_job: last_job=cur_job
        if tag=="dischema":     cur_schema=(a.get("name") or a.get("displayname") or cur_schema).strip()

        # ---------- sources / targets (DB) ----------
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

        # ---------- file sources / targets ----------
        if tag in ("difilesource","difiletarget"):
            proj,job,df=context_for(e)
            fmt=a.get("formatname") or ""
            fname=a.get("filename") or ""
            ds=a.get("datastore") or a.get("datastorename") or "FILE"
            sch=fmt or "FILE"
            tbl=fname or (a.get("name") or "")
            remember_display(ds,sch,tbl)
            role="source" if "source" in tag else "target"
            key=(proj,job,df,role,_norm_key(ds),_norm_key(sch),_norm_key(tbl))
            source_target.add(key)

        # ---------- standard lookup (column level) ----------
        if tag=="diattribute" and lower(a.get("name",""))=="ui_mapping_text":
            txt=a.get("value") or e.text or ""
            if HAS_LOOKUP.search(txt):
                proj,job,df=context_for(e)
                schema_out=schema_out_from_DISchema(e, pm, cur_schema)
                col=find_output_column(e, pm)
                dsl,schl,tbl=extract_lookup_from_text(txt)
                if dsl and tbl and schema_out and col:
                    remember_display(dsl,schl,tbl)
                    lookup_pos[(proj,job,df,_norm_key(dsl),_norm_key(schl),_norm_key(tbl))]\
                        .append(f"{schema_out}>>{col}")

        # ---------- lookup_ext ONLY from FUNCTION_CALL ----------
        if tag=="function_call" and lower(a.get("name",""))=="lookup_ext":
            proj,job,df=context_for(e)
            schema_out=schema_out_from_DISchema(e, pm, cur_schema)
            dsx,schx,tbx = extract_lookup_ext_from_function_call(collect_text(e), a)
            if dsx and tbx and schema_out:
                remember_display(dsx,schx,tbx)
                k = (proj,job,df,_norm_key(dsx),_norm_key(schx),_norm_key(tbx))
                lookup_ext_pos[k].add(schema_out)

        # ---------- custom SQL ----------
        if tag=="ditransformcall" and (a.get("typeid")=="24" or lower(a.get("type") or "")=="sql"):
            proj,job,df=context_for(e)

            # name of the SQL block for labeling
            ui_disp=""
            for ch in e.iter():
                at=attrs_ci(ch)
                if lower(strip_ns(getattr(ch,"tag","")))=="diattribute" and lower(at.get("name",""))=="ui_display_name":
                    ui_disp=at.get("value") or ui_disp

            # SQL text
            sql_txt=""
            for ch in e.iter():
                if lower(strip_ns(getattr(ch,"tag","")))=="sqltext":
                    at=attrs_ci(ch)
                    sql_txt = at.get("sql_text") or ch.text or sql_txt
                    break

            sql_txt = (sql_txt or "").strip()
            # escape backslashes outside an f-string context
            safe_sql_txt = sql_txt.replace("\\", "\\\\")
            # find table-like tokens for a quick table list
            tables = []
            for m in SQL_TABLE_RE.finditer(sql_txt):
                val = m.group(1).strip().strip('"')
                if val: tables.append(val)
            tables = dedupe([t for t in tables if t])

            # we emit one synthesized row describing the SQL transform itself
            sql_rows.append((
                proj or "", job or "", df or "",
                "source",               # treat as a source-like transform
                "DS_SQL" ,              # synthetic datastore label
                "CUSTOM_SQL",           # schema marker
                ", ".join(tables) or (ui_disp or "SQL"),  # table field shows table list or name
                ui_disp,                # transformation_position: keep the display name here
                len(tables) if tables else 1,
                f"\"{safe_sql_txt}\""   # explicitly quoted text
            ))

    # ---------- finalize rows ----------

    def nice_names(dsN, schN, tblN):
        k=(dsN, schN, tblN)
        return ( display_ds[k].get(dsN), display_sch[k].get(schN), display_tbl[k].get(tblN) )

    Row = namedtuple("Row", [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table","transformation_position",
        "transformation_usages_count","custom_sql_text"
    ])

    rows=[]

    # lookups
    for (proj,job,df,dsN,schN,tblN), positions in lookup_pos.items():
        uniq=sorted(dedupe([p.strip() for p in positions if p and p.strip()]))
        if not uniq: continue
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        rows.append(Row(proj or "", job or "", df or "", "lookup",
                        dsD, schD, tblD, ", ".join(uniq), len(uniq), ""))

    # lookup_ext
    for (proj,job,df,dsN,schN,tblN), posset in lookup_ext_pos.items():
        uniq=sorted(dedupe([p.strip() for p in posset if p and p.strip()]))
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        rows.append(Row(proj or "", job or "", df or "", "lookup_ext",
                        dsD, schD, tblD, ", ".join(uniq), len(uniq), ""))

    # sources / targets
    for (proj,job,df,role,dsN,schN,tblN) in sorted(source_target):
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        rows.append(Row(proj or "", job or "", df or "", role,
                        dsD, schD, tblD, "", 0, ""))

    # custom SQL rows
    for r in sql_rows:
        rows.append(Row(*r))

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # strict dedupe by normalized key and merge positions & counts
    def nkey(r):
        return (r["project_name"], r["job_name"], r["dataflow_name"], r["role"],
                _norm_key(r["datastore"]), _norm_key(r["schema"]), _norm_key(r["table"]))

    df["__k__"]=df.apply(nkey, axis=1)
    agg = (df.groupby(["__k__","project_name","job_name","dataflow_name","role",
                       "datastore","schema","table"], dropna=False, as_index=False)
             .agg({
                 "transformation_position": lambda s: ", ".join(sorted(dedupe([x.strip() for x in s if str(x).strip()]))),
                 "transformation_usages_count": "sum",
                 "custom_sql_text": lambda s: ", ".join([x for x in dedupe([str(x) for x in s if str(x).strip()])])
             }))

    df = agg.drop(columns=["__k__"])

    # final display cleanup
    for c in ("datastore","schema","table"): df[c]=df[c].map(_pretty)

    df=df.sort_values(by=["project_name","job_name","dataflow_name","role","datastore","schema","table"]).reset_index(drop=True)
    return df

# ------------------------ main ------------------------

def main():
    # ---- set these paths ----
    xml_path = r"C:\path\to\export.xml"                     # your SAP DS export (project/job/df)
    out_xlsx = r"C:\path\to\output_v9.xlsx"                 # output Excel

    df = parse_single_xml(xml_path)

    cols = [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table",
        "transformation_position","transformation_usages_count","custom_sql_text"
    ]
    for c in cols:
        if c not in df.columns: df[c]=""
    df = df[cols]

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print(f"Done. Wrote: {out_xlsx}  |  Rows: {len(df)}")

if __name__ == "__main__":
    main()
