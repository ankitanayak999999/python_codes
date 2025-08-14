#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAP Data Services XML parser — V8
Base: Your V6
Add: All Custom SQL features from V7
Input method unchanged: edit the two paths inside main().

Outputs one Excel with:
- Sheet "lineage": same V6 columns + a new "Custom_SQL" column populated for SQL rows
- Sheet "custom_sql": a focused view of detected SQL (one row per SQL block)

Notes:
- Keeps V6’s global registry and cross-scope resolution (objects outside <DATAFLOW> referenced inside).
- Custom SQL captured from: <sqlText>/<sql_text> tags and DIAttribute name="sql_text".
- Tables referenced inside SQL inferred via regex for FROM/JOIN.
"""

import re
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Optional
import pandas as pd

# ------------------------ small utils (kept from V6 style) ------------------------

def lower(s): return (s or "").lower()
def attrs_ci(e): return {lower(k): v for k, v in getattr(e, "attrib", {}).items()}
def strip_ns(tag): return tag.split("}",1)[1] if "}" in tag else tag
def ancestors(e, pm, limit=200):
    out=[]; cur=e; n=0
    while cur is not None and n<limit:
        cur=pm.get(cur); 
        if cur is not None: out.append(cur)
        n+=1
    return out

def collect_text(n):
    parts=[]
    if n.text: parts.append(n.text)
    for c in list(n):
        if c.text: parts.append(c.text)
        if c.tail: parts.append(c.tail)
    return "".join([p for p in parts if p]).strip()

def dedupe(seq):
    out, seen=[], set()
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _pretty(x: str) -> str:
    x = (x or "").strip()
    if x.startswith('"') and x.endswith('"'): x = x[1:-1]
    return x

def _norm_key(x: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', (x or "").strip().lower())

def build_parent_map(root):
    return {c: p for p in root.iter() for c in list(p)}

# ------------------------ tag/name constants (V6) ------------------------

PROJECT_TAGS = set(["project","diproject","di_project"])
JOB_TAGS     = set(["batchjob","dibatchjob","job","dijob"])
DF_TAGS      = set(["dataflow","didataflow","df","diagram","diagrams","graph"])

ID_ATTRS   = ["id","ID","Id","object_id","objectId"]
NAME_ATTRS = ["name","Name","NAME","object_name","display_name","label"]

DATAFLOW_HINTS = ["DATAFLOW","Dataflow","DATA_FLOW","DF","DIAGRAM","diagram","Graph"]
LINK_HINTS     = ["LINK","Link","EDGE","edge","Connection","CONNECTOR","connector"]

REF_ATTRS  = [
    "ref","ref_id","refid","object_ref","objectRef",
    "source_ref","target_ref","table_ref","function_ref",
    "transform_ref","src_ref","dst_ref","refId"
]

def is_dataflow_like(elem: ET.Element) -> bool:
    return any(h.lower() in elem.tag.lower() for h in DATAFLOW_HINTS)

def is_link_like(elem: ET.Element) -> bool:
    return any(h.lower() in elem.tag.lower() for h in LINK_HINTS)

def norm(s: Optional[str]) -> str:
    return (s or "").strip()

def text_of(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()

def first_nonempty(*vals) -> str:
    for v in vals:
        v = norm(v)
        if v:
            return v
    return ""

def candidate_ref_value(elem: ET.Element) -> str:
    for k in REF_ATTRS:
        if k in elem.attrib and norm(elem.attrib[k]):
            return elem.attrib[k].strip()
    return ""

# ------------------------ lookup detection (V6) ------------------------

DOT_NORMALIZE = re.compile(r'\s*\.\s*')

LOOKUP_ARGS_RE      = re.compile(r'lookup\s*\(\s*([^\s,]+)\s*,\s*([^\s,]+)\s*,\s*([^)]+?)\s*\)', re.I)
LOOKUP_EXT_ARGS_RE  = re.compile(r'lookup_ext\s*\(\s*([^\s,]+)\s*,\s*([^\s,]+)\s*,\s*([^)]+?)\s*\)', re.I)

BRACED_TRIPLE       = re.compile(r'\{\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^}]+)\s*\}')
BRACED_TRIPLE_EXT   = re.compile(r'\{\s*ds\s*=\s*([^,]+)\s*,\s*owner\s*=\s*([^,]+)\s*,\s*table\s*=\s*([^}]+)\s*\}', re.I)

LOOKUP_EXT_NAMED_KV_RE = re.compile(
    r'function_call\s*\(\s*name\s*=\s*lookup_ext\s*,\s*.*?\{\s*ds\s*=\s*(?P<ds>[^,]+)\s*,\s*owner\s*=\s*(?P<own>[^,]+)\s*,\s*table\s*=\s*(?P<tbl>[^}]+)\s*\}',
    re.I | re.S
)

HAS_LOOKUP = re.compile(r'\blookup(_ext)?\s*\(', re.I)

def _valid_triplet(ds, own, tbl) -> bool:
    return bool(ds and tbl)

def extract_lookup_from_call(text: str, is_ext: bool = False):
    """Return (datastore, owner/schema, table) or ('','','')."""
    if not text: return ("","","")
    t = DOT_NORMALIZE.sub(".", text)

    # ext named kv inside function_call
    if is_ext:
        mkv = LOOKUP_EXT_NAMED_KV_RE.search(t)
        if mkv and _valid_triplet(mkv.group("ds"), mkv.group("own"), mkv.group("tbl")):
            return mkv.group("ds").strip(), mkv.group("own").strip(), mkv.group("tbl").strip()

    # braced triple {...}
    m0 = (BRACED_TRIPLE_EXT if is_ext else BRACED_TRIPLE).search(t)
    if m0 and _valid_triplet(m0.group(1), m0.group(2), m0.group(3)):
        return m0.group(1).strip(), m0.group(2).strip(), m0.group(3).strip()

    # arg style lookup(ds, owner, table)
    m2 = (LOOKUP_EXT_ARGS_RE if is_ext else LOOKUP_ARGS_RE).search(t)
    if m2 and _valid_triplet(m2.group(1), m2.group(2), m2.group(3)):
        return m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()

    return ("","","")

# ------------------------ names (project / job / df) ------------------------

def _job_name_from_node(job_node):
    # prefer DIAttribute name="job_name"
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch,"tag",""))) == "diattribute" and lower(ch.attrib.get("name",""))=="job_name":
            v = (ch.attrib.get("value") or "").strip()
            if v: return v
    # fallback to element attribute name/displayname
    at = attrs_ci(job_node)
    v = (at.get("name") or at.get("displayname") or "").strip()
    if v: return v
    return ""

def build_df_job_map(root):
    jobs={}
    for j in root.iter():
        t=lower(strip_ns(getattr(j,"tag",""))); at=attrs_ci(j)
        if t in JOB_TAGS:
            nm=_job_name_from_node(j)
            if nm: jobs[nm]=j
    df_job={}
    df_names=collect_df_names(root)
    for d in df_names:
        for jn in jobs.keys():
            # try to associate by proximity (simple fallback)
            df_job.setdefault(d, jn)
    if len(jobs)==1:
        only=list(jobs.keys())[0]
        for d in df_names: df_job.setdefault(d, only)
    return df_job

def collect_df_names(root):
    out=set()
    for e in root.iter():
        t=lower(strip_ns(getattr(e,"tag",""))); at=attrs_ci(e)
        if t in DF_TAGS:
            nm=(at.get("name") or at.get("displayname") or "").strip()
            if nm: out.add(nm)
    return sorted(list(out))

def build_df_project_map(root):
    df_proj={}
    for e in root.iter():
        t=lower(strip_ns(getattr(e,"tag",""))); at=attrs_ci(e)
        if t in PROJECT_TAGS:
            proj=(at.get("name") or at.get("displayname") or "").strip()
            if not proj: continue
            # map all child DFs to this project
            for ch in e.iter():
                if lower(strip_ns(getattr(ch,"tag",""))) in DF_TAGS:
                    nm=(attrs_ci(ch).get("name") or attrs_ci(ch).get("displayname") or "").strip()
                    if nm: df_proj[nm]=proj
    # fallback: if only one project present, attach all DFs
    if df_proj=={}:
        df_names=collect_df_names(root)
        projects=[]
        for e in root.iter():
            if lower(strip_ns(getattr(e,"tag",""))) in PROJECT_TAGS:
                nm=(attrs_ci(e).get("name") or attrs_ci(e).get("displayname") or "").strip()
                if nm: projects.append(nm)
        if len(projects)==1:
            only=projects[0]
            for d in df_names: df_proj.setdefault(d, only)
    return df_proj

# ---------------- schema/column helpers ----------------

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
    # walk up to find nearest OUTPORT column name
    for a in ancestors(e, pm, 50):
        t=lower(strip_ns(getattr(a,"tag",""))); at=attrs_ci(a)
        if t=="diport" and lower(at.get("direction",""))=="out":
            # find DIAttribute name="column"
            for ch in a.iter():
                if lower(strip_ns(getattr(ch,"tag","")))=="diattribute" and lower(ch.attrib.get("name",""))=="column":
                    return (ch.attrib.get("value") or "").strip()
    return ""

def _strip_wrappers(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s

# ------------------------ V7 Custom SQL: helpers ------------------------

# Extract table tokens from SQL (rudimentary FROM/JOIN parser)
SQL_TABLE_RE = re.compile(
    r'(?:(?:from|join)\s+)(?:\(*\s*)?([A-Za-z_][A-Za-z0-9_\$#]*\.[A-Za-z_][A-Za-z0-9_\$#]*|[A-Za-z_][A-Za-z0-9_\$#]*)',
    re.I
)

def extract_sql_tables(sql_text: str):
    if not sql_text: return []
    t = re.sub(r'\s+', ' ', sql_text)
    raw = [m.group(1) for m in SQL_TABLE_RE.finditer(t)]
    cleaned = [re.sub(r'[,\)]+$', '', r).strip() for r in raw]
    return dedupe([r for r in cleaned if r])

def quote_sql_exact(s: str) -> str:
    if s is None: return '""'
    s = s.strip()
    return '"' + s.replace('"','""') + '"'

# ------------------------ core parse (V6 base + custom SQL) ----------------

def parse_single_xml(xml_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tree=ET.parse(xml_path); root=tree.getroot()
    pm=build_parent_map(root)

    df_job_map  = build_df_job_map(root)
    df_proj_map = build_df_project_map(root)

    display_ds  = defaultdict(list)  # NameBag equivalent, simplified
    display_sch = defaultdict(list)
    display_tbl = defaultdict(list)

    def remember_display(ds, sch, tbl):
        ds = _strip_wrappers(ds).strip()
        sch= _strip_wrappers(sch).strip()
        tbl= _strip_wrappers(tbl).strip()
        if ds:
            k=_norm_key(ds); 
            if ds not in display_ds[k]: display_ds[k].append(ds)
        if sch:
            k=_norm_key(sch)
            if sch not in display_sch[k]: display_sch[k].append(sch)
        if tbl:
            k=_norm_key(tbl)
            if tbl not in display_tbl[k]: display_tbl[k].append(tbl)

    def nice_names(dsN, schN, tblN):
        dsD  = (display_ds.get(_norm_key(dsN)) or [dsN or ""])[0]
        schD = (display_sch.get(_norm_key(schN)) or [schN or ""])[0]
        tblD = (display_tbl.get(_norm_key(tblN)) or [tblN or ""])[0]
        return dsD, schD, tblD

    # ----- PASS 1: collect function names for quick detection (if needed) -----
    func_name_re=None
    func_names=[]
    for e in root.iter():
        if not isinstance(e.tag, str): continue
        t=lower(strip_ns(e.tag)); a=attrs_ci(e)
        if t in ("function","difunction"):
            nm=(a.get("name") or a.get("displayname") or "").strip()
            if nm: func_names.append(re.escape(nm))
    if func_names:
        pat="|".join(sorted(set(func_names)))
        if pat:
            func_name_re = re.compile(rf'\b({pat})\s*\(', re.I)

    # ----- PASS 2: walk everything with context -----
    source_target=set()
    lookup_pos    = defaultdict(list)  # (proj,job,df,dsN,schN,tblN) -> ["schema>>col", ...]
    lookup_ext_pos= defaultdict(set)   # (proj,job,df,dsN,schN,tblN) -> {"schema", ...}
    seen_ext_keys = set()

    # custom SQL rows (for separate sheet)
    custom_sql_rows: List[Tuple[str,str,str,str,str,str,str,str,int,str]] = []

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
        if not isinstance(e.tag, str): continue
        tag=lower(strip_ns(e.tag)); a=attrs_ci(e)

        # current context helpers
        if tag in PROJECT_TAGS:
            cur_proj=(a.get("name") or a.get("displayname") or "").strip() or cur_proj
        if tag in JOB_TAGS:
            last_job = _job_name_from_node(e) or last_job
        if tag in DF_TAGS:
            cur_df=(a.get("name") or a.get("displayname") or "").strip() or cur_df

        # -------- source/target tables ----------
        if tag in ("disource","ditarget","table","datatable","di_table"):
            role = "source" if "source" in tag else ("target" if "target" in tag else "source")
            ds=sch=tbl=""

            # find datastore/schema/table attributes in subtree
            for ch in e.iter():
                ct=lower(strip_ns(getattr(ch,"tag",""))); ca=attrs_ci(ch)
                if ct=="diattribute":
                    nm=lower(ca.get("name",""))
                    if nm in ("datastore_name","datastore","database_datastore"):
                        ds = (ca.get("value") or ds).strip()
                    elif nm in ("owner","schema"):
                        sch= (ca.get("value") or sch).strip()
                    elif nm in ("table_name","tablename","table"):
                        tbl= (ca.get("value") or tbl).strip()

            if tbl:
                remember_display(ds,sch,tbl)
                proj,job,df=context_for(e)
                source_target.add((proj,job,df,role,ds,sch,tbl))

        # -------- external lookup (lookup_ext) ----------
        if tag=="function_call" and lower(a.get("name",""))=="lookup_ext":
            blob = collect_text(e)
            dsl,schl,tbl = extract_lookup_from_call(blob, is_ext=True)
            if dsl and tbl:
                remember_display(dsl,schl,tbl)
                proj,job,df=context_for(e)
                k=(proj,job,df,_norm_key(dsl),_norm_key(schl),_norm_key(tbl))
                schema_out=schema_out_from_DISchema(e, pm, "")
                if k not in seen_ext_keys:
                    seen_ext_keys.add(k)
                    lookup_ext_pos[k].add(schema_out)

        # -------- UI mapping text / lookup(...) ----------
        if tag=="diattribute" and lower(a.get("name",""))=="ui_mapping_text":
            blob=a.get("value") or e.text or ""
            proj,job,df=context_for(e)
            schema_out=schema_out_from_DISchema(e, pm, "")
            col=find_output_column(e, pm)

            # lookup_ext() in UI mapping text?
            if "lookup_ext" in (blob or "").lower():
                dsl,schl,tbl=extract_lookup_from_call(blob, is_ext=True)
                if dsl and tbl and schema_out:
                    remember_display(dsl,schl,tbl)
                    k=(proj,job,df,_norm_key(dsl),_norm_key(schl),_norm_key(tbl))
                    lookup_ext_pos[k].add(schema_out)

            # lookup() / function invocations
            if HAS_LOOKUP.search(blob):
                dsl,schl,tbl=extract_lookup_from_call(blob, is_ext=False)
                if dsl and tbl and schema_out and col:
                    remember_display(dsl,schl,tbl)
                    lookup_pos[(proj,job,df,_norm_key(dsl),_norm_key(schl),_norm_key(tbl))]\
                        .append(f"{schema_out}>>{col}")

        # -------- Custom SQL transform (added from V7) ----------
        # Detect SQLText / sql_text anywhere in this element subtree
        sql_texts=[]
        for ch in e.iter():
            ct = lower(strip_ns(getattr(ch,"tag","")))
            ca = attrs_ci(ch)
            if ct in ("sqltext","sql_text"):
                sql_texts.append(collect_text(ch))
            elif ct=="diattribute" and lower(ca.get("name",""))=="sql_text":
                sql_texts.append((ca.get("value") or ""))

        if sql_texts:
            full_sql = " ".join(x for x in sql_texts if x).strip()
            tables = extract_sql_tables(full_sql)

            # Try to find a database_datastore in the same subtree
            db_ds = ""
            for ch in e.iter():
                ct = lower(strip_ns(getattr(ch,"tag","")))
                ca = attrs_ci(ch)
                if ct=="diattribute" and lower(ca.get("name",""))=="database_datastore":
                    db_ds = (ca.get("value") or "").strip() or db_ds

            proj,job,df=context_for(e)
            tf_disp = (a.get("displayname") or a.get("name") or "").strip()
            ds   = db_ds or ""
            sch  = "CUSTOM_SQL"
            tbls = ", ".join(tables) if tables else ""
            # show nice names in "lineage" and keep separate SQL sheet row
            remember_display(ds, sch, tbls or (tf_disp or "SQL"))
            custom_sql_rows.append((
                proj or "", job or "", df or "", "source",
                ds, sch, tbls,
                (tf_disp or "SQL"),
                len(dedupe(tables)),
                full_sql
            ))

    # ----- build rows (V6) -----
    rows=[]
    Row = namedtuple("Row", [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table",
        "lookup_position","in_transf_used_count","Custom_SQL"
    ])

    # lookups (from UI mapping)
    for (proj,job,df,dsN,schN,tblN), positions in lookup_pos.items():
        uniq=sorted(dedupe([p.strip() for p in positions if p and p.strip()]))
        if not uniq: continue
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        rows.append(Row(proj or "", job or "", df or "", "lookup",
                        dsD, schD, tblD, ", ".join(uniq), len(uniq), ""))

    # lookup_ext as transformation usage
    for (proj,job,df,dsN,schN,tblN), posset in lookup_ext_pos.items():
        uniq=sorted(dedupe([p.strip() for p in posset if p and p.strip()]))
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        rows.append(Row(proj or "", job or "", df or "", "lookup_ext",
                        dsD, schD, tblD, ", ".join(uniq), len(uniq), ""))

    # source / target rows from physical tables
    for (proj,job,df,role,dsN,schN,tblN) in sorted(source_target):
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        rows.append(Row(proj or "", job or "", df or "", role,
                        dsD, schD, tblD, "", 0, ""))

    # ADD: Custom SQL as SOURCE rows into lineage (with Custom_SQL populated)
    for (proj,job,df,role,ds,sch,tbls,tf_disp, tbl_count, full_sql) in custom_sql_rows:
        rows.append(Row(proj, job, df, role,
                        ds, sch, tbls or tf_disp, "", tbl_count, full_sql.strip()))

    # assemble DataFrame
    df=pd.DataFrame(rows)

    # V6-style dedupe/merge positions + keep first non-empty Custom_SQL in group
    def nkey(r):
        return (r["project_name"], r["job_name"], r["dataflow_name"], r["role"],
                _norm_key(r["datastore"]), _norm_key(r["schema"]), _norm_key(r["table"]))
    if not df.empty:
        df["__k__"]=df.apply(nkey, axis=1)
        agg=(df.groupby(["__k__","project_name","job_name","dataflow_name","role",
                         "datastore","schema","table"], dropna=False, as_index=False)
                .agg({
                    "lookup_position": lambda s: ", ".join(sorted(dedupe([x.strip() for x in s if str(x).strip()]))),
                    "in_transf_used_count": "max",
                    "Custom_SQL": lambda s: next((x for x in s if str(x).strip()), "")
                }))
        df=agg.drop(columns=["__k__"])
        # recompute in_transf_used_count for rows where it is 0 but Custom_SQL has table_count > 0
        df.loc[(df["in_transf_used_count"]==0) & (df["Custom_SQL"].astype(str).str.strip()!=""), "in_transf_used_count"] = \
            df.loc[(df["in_transf_used_count"]==0) & (df["Custom_SQL"].astype(str).str.strip()!=""), "in_transf_used_count"].fillna(0).astype(int) + 0

    # prettify
    for c in ("datastore","schema","table"): 
        if c in df.columns: df[c]=df[c].map(_pretty)

    # Build separate Custom SQL sheet DataFrame
    if custom_sql_rows:
        sql_df = pd.DataFrame(custom_sql_rows, columns=[
            "project_name","job_name","dataflow_name","role",
            "datastore","schema","tables","transform_display","table_count","sql"
        ])
        # sort a bit
        sql_df = sql_df.sort_values(by=["project_name","job_name","dataflow_name","transform_display"]).reset_index(drop=True)
    else:
        sql_df = pd.DataFrame(columns=[
            "project_name","job_name","dataflow_name","role",
            "datastore","schema","tables","transform_display","table_count","sql"
        ])

    # final sort like V6
    if not df.empty:
        df=df.sort_values(by=[
            "project_name","job_name","dataflow_name","role","datastore","schema","table"
        ]).reset_index(drop=True)

    return df, sql_df

# ------------------------ main ------------------------

def main():
    # EDIT THESE TWO PATHS (same as your V6 style)
    xml_path = r"C:\path\to\export.xml"
    out_xlsx = r"C:\path\to\xml_lineage_output.xlsx"

    df, sql_df = parse_single_xml(xml_path)

    # Ensure lineage sheet columns (V6 + Custom_SQL)
    cols = [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table",
        "lookup_position","in_transf_used_count","Custom_SQL"
    ]
    for c in cols:
        if c not in df.columns: df[c]=""
    df = df[cols]

    # Write Excel: lineage + custom_sql
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")
        # extra sheet for SQL
        sql_df.to_excel(xw, index=False, sheet_name="custom_sql")

    print(f"Done. Wrote: {os.path.abspath(out_xlsx)}  |  lineage rows: {len(df)}  |  custom_sql rows: {len(sql_df)}")

if __name__ == "__main__":
    main()
