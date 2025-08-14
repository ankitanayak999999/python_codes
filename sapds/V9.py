#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V9: SAP DS export XML lineage extractor

Keeps V8 behavior and:
- lookup: captured from UI mapping text / generic blobs (unchanged)
- lookup_ext: ONLY from <FUNCTION_CALL name="lookup_ext" ...> (no DIExpression fallback)
- Custom SQL: STRICT like V8 — only when a real <sql_text> exists and is non-empty.
  schema/table derived from FROM/JOIN; SQL text written in double quotes.
- File sources/targets supported (schema=format name, table=file name, datastore=FILE if missing)
- Multi-project export: job/workflow → dataflow mapping, and project mapping
- De-duplication with merged transformation positions
- Columns:
  project_name, job_name, dataflow_name, role,
  datastore, schema, table,
  transformation_position, transformation_usages_count, custom_sql_text
"""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from typing import Dict, Iterable
import pandas as pd

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

NAME_CHARS    = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"
DOT_NORMALIZE = re.compile(r"\s*\.\s*")

# detect presence
HAS_LOOKUP     = re.compile(r'\blookup(?!_)\s*\(', re.I)

# standard lookup parsers
LOOKUP_CALL_RE     = re.compile(rf'\blookup(?!_)\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)
BRACED_TRIPLE      = re.compile(r'\blookup(?!_)\s*\(\s*\{\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^\}]+?)\s*\}', re.I|re.S)
LOOKUP_ARGS_RE     = re.compile(r'\blookup(?!_)\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*(?:,|\))', re.I|re.S)

# lookup_ext from FUNCTION_CALL only
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
    a_str = " ".join([f'{k}="{v}"' for k,v in attrs.items()])
    m = LOOKUP_EXT_NAMED_KV_RE.search(a_str)
    if m and _valid_triplet(m.group("ds"), m.group("own"), m.group("tbl")):
        return m.group("ds").strip(), m.group("own").strip(), m.group("tbl").strip()

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
    rows_custom_sql = []               # temp rows for custom SQL

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
            role="source" if "source" in tag else "target"
            fmt = a.get("formatname","").strip()
            fname = a.get("filename","").strip()
            ds = a.get("datastorename","").strip() or a.get("datastore","").strip() or "FILE"
            sch = fmt or "FILE"
            tbl = fname or (a.get("name","").strip() or "FILE")
            remember_display(ds,sch,tbl)
            key=(proj,job,df,role,_norm_key(ds),_norm_key(sch),_norm_key(tbl))
            source_target.add(key)

        # ---------- standard lookup (column level via UI mapping) ----------
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

        # ---------- Custom SQL (STRICT like V8: only when <sql_text> exists) ----------
        if tag == "ditransformcall" and (a.get("typeid") == "24" or lower(a.get("type") or "") == "sql"):
            proj, job, df = context_for(e)

            # Optional friendly name shown in canvas
            ui_disp = ""
            for ch in e.iter():
                at2 = attrs_ci(ch)
                if lower(strip_ns(getattr(ch, "tag", ""))) == "diattribute" and lower(at2.get("name","")) == "ui_display_name":
                    ui_disp = at2.get("value") or ui_disp

            # STRICT source: read only from <SQLText><sql_text>…</sql_text>
            sql_text = ""
            for ch in e.iter():
                if lower(strip_ns(getattr(ch, "tag", ""))) == "sqltext":
                    for k in ch.iter():
                        if lower(strip_ns(getattr(k, "tag", ""))) == "sql_text":
                            sql_text = (k.text or "").strip()
                            break
                    if sql_text:
                        break

            # If no proper sql_text, SKIP (prevents false custom SQL rows)
            if not sql_text:
                pass
            else:
                # owner.table tokens from FROM / JOIN (quick parse)
                tables = []
                for m in SQL_TABLE_RE.finditer(sql_text):
                    val = m.group(1).strip().strip('"')
                    if val: tables.append(val)
                tables = dedupe([t for t in tables if t])

                # schema/table fields
                if tables:
                    schemas = ", ".join(sorted(dedupe([t.split(".")[0] if "." in t else t for t in tables])))
                    tablenames = ", ".join(sorted(dedupe([t.split(".")[-1] for t in tables])))
                else:
                    schemas = "CUSTOM_SQL"
                    tablenames = ui_disp or "SQL"

                # SQL as one line, quoted; avoid f-string backslash issues
                safe_sql_text = " ".join(sql_text.split()).replace('"', '""')
                custom_sql_col = "\"" + safe_sql_text + "\""

                rows_custom_sql.append((
                    proj or "", job or "", df or "",
                    "source",
                    "",  # keep datastore blank, like V8
                    schemas,
                    tablenames,
                    ui_disp or "SQL",
                    max(1, len(tables)),
                    custom_sql_col
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
    for r in rows_custom_sql:
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
    # Set these paths before running:
    xml_path = r"C:\path\to\export.xml"
    out_xlsx = r"C:\path\to\output_v9.xlsx"

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

