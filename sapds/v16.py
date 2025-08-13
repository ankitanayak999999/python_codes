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
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def _strip_wrappers(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().strip('"').strip("'")
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        s = s[1:-1]
    return s

def _pretty(s: str) -> str:
    s = _strip_wrappers(s)
    s = re.sub(r"[\{\}\[\]]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_key(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', _pretty(s).upper())

# ------------------------ patterns & detectors ------------------------
NAME_CHARS = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"
DOTS = re.compile(r"\s*\.\s*")

# cache/policy tokens to ignore in triples
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

def _valid(ds, sch, tbl): return bool(ds) and _looks_like_table(tbl) and not _is_cache_or_policy(sch)

# lookup/lookup_ext detection & extraction (attributes + dotted + arg triples)
HAS_LOOKUP     = re.compile(r'\blookup(?!_)\s*\(', re.I)
HAS_LOOKUP_EXT = re.compile(r'\blookup[_ ]?ext\s*\(', re.I)

LOOKUP_ARGS     = re.compile(r'\blookup(?!_)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,\)]+)', re.I|re.S)
LOOKUP_DOTTED   = re.compile(rf'\blookup(?!_)\s*\(\s*"?({NAME_CHARS})"?\s*\.\s*"?({NAME_CHARS})"?\s*\.\s*"?({NAME_CHARS})"?', re.I)

LOOKUP_EXT_ARGS   = re.compile(r'\blookup[_ ]?ext\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,\)]+)', re.I|re.S)
LOOKUP_EXT_DOTTED = re.compile(rf'\blookup[_ ]?ext\s*\(\s*"?({NAME_CHARS})"?\s*\.\s*"?({NAME_CHARS})"?\s*\.\s*"?({NAME_CHARS})"?', re.I)
LOOKUP_EXT_KV     = re.compile(
    r'\blookup[_ ]?ext\s*\([^)]*?'
    r'(?:tableDatastore|tabledatastore|tabledatastorename|datastorename)\s*=\s*([\'"]?)(?P<ds>[^\'",)\s]+)\1[^)]*?'
    r'(?:tableOwner|tableowner|ownername)\s*=\s*([\'"]?)(?P<own>[^\'",)\s]+)\3[^)]*?'
    r'(?:tableName|tablename|name)\s*=\s*([\'"]?)(?P<tbl>[^\'",)\s]+)\5',
    re.I|re.S
)

def extract_lookup_triple(text: str, is_ext: bool):
    if not text: return ("","","")
    t = DOTS.sub(".", text)

    if is_ext:
        mkv = LOOKUP_EXT_KV.search(t)
        if mkv and _valid(mkv.group("ds"), mkv.group("own"), mkv.group("tbl")):
            return mkv.group("ds").strip(), mkv.group("own").strip(), mkv.group("tbl").strip()

    m = (LOOKUP_EXT_DOTTED if is_ext else LOOKUP_DOTTED).search(t)
    if m and _valid(m.group(1), m.group(2), m.group(3)):
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    m = (LOOKUP_EXT_ARGS if is_ext else LOOKUP_ARGS).search(t)
    if m and _valid(m.group(1), m.group(2), m.group(3)):
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    return ("","","")

# ------------------------ high-level tags ------------------------
DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")

# ------------------------ context helpers ------------------------
def job_name_from_node(job_node):
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch,"tag",""))) == "diattribute" and lower(ch.attrib.get("name",""))=="job_name":
            v = (ch.attrib.get("value") or "").strip()
            if v: return v
    return (job_node.attrib.get("name") or job_node.attrib.get("displayName") or "").strip()

def map_df_to_project(root):
    out={}
    projects=[]
    for p in root.iter():
        if lower(strip_ns(getattr(p,"tag",""))) in PROJECT_TAGS:
            nm=(p.attrib.get("name") or p.attrib.get("displayName") or "").strip()
            if nm: projects.append((nm,p))
    for proj_name, pnode in projects:
        for d in pnode.iter():
            if lower(strip_ns(getattr(d,"tag",""))) in DF_TAGS:
                dn=(d.attrib.get("name") or d.attrib.get("displayName") or "").strip()
                if dn: out.setdefault(dn, proj_name)
    if len(projects)==1:
        only=projects[0][0]
        for d in root.iter():
            if lower(strip_ns(getattr(d,"tag",""))) in DF_TAGS:
                dn=(d.attrib.get("name") or d.attrib.get("displayName") or "").strip()
                if dn: out.setdefault(dn, only)
    return out

def map_df_to_job(root):
    # simple heuristic: nearest ancestor job/workflow that references the DF
    pm=build_parent_map(root)
    df_names=set()
    for n in root.iter():
        if lower(strip_ns(getattr(n,"tag",""))) in DF_TAGS:
            nm=(n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: df_names.add(nm)
    jobs=[]
    for n in root.iter():
        if lower(strip_ns(getattr(n,"tag",""))) in JOB_TAGS:
            jobs.append((job_name_from_node(n), n))
    out={}
    # direct containment
    for df in root.iter():
        if lower(strip_ns(getattr(df,"tag",""))) in DF_TAGS:
            dn=(df.attrib.get("name") or df.attrib.get("displayName") or "").strip()
            cur=pm.get(df)
            while cur is not None:
                t=lower(strip_ns(cur.tag))
                if t in JOB_TAGS:
                    out.setdefault(dn, job_name_from_node(cur)); break
                cur=pm.get(cur)
    if len(jobs)==1:
        only=jobs[0][0]
        for dn in df_names: out.setdefault(dn, only)
    return out

def nearest_schema_out(elem, pm, fallback=""):
    best=None; join=None
    for a in ancestors(elem, pm, 200):
        if lower(strip_ns(getattr(a,"tag","")))=="dischema":
            nm=(attrs_ci(a).get("name") or "").strip()
            if nm and lower(nm)!="join":
                best=nm; break
            elif nm: join=nm
    return best or join or fallback

def nearest_output_col(elem, pm):
    for a in ancestors(elem, pm, 60):
        if lower(strip_ns(getattr(a,"tag","")))=="dielement":
            nm=(attrs_ci(a).get("name") or "").strip()
            if nm: return nm
    return ""

# ------------------------ main parser ------------------------
Record = namedtuple("Record", [
    "project_name","job_name","dataflow_name",
    "role","datastore","schema","table",
    "lookup_position","in_transf_used_count"
])

def parse_single_xml(xml_path: str):
    tree=ET.parse(xml_path); root=tree.getroot()
    pm=build_parent_map(root)

    df_proj = map_df_to_project(root)
    df_job  = map_df_to_job(root)

    source_target=set()
    lookup_pos   = defaultdict(list)  # (proj,job,df,ds,sch,tbl) -> ["Schema>>Col", ...]
    lookup_ext_s = defaultdict(set)   # (proj,job,df,ds,sch,tbl) -> {"SchemaA","SchemaB"...}
    seen_ext_fun = set()              # keys captured via FUNCTION_CALL (to avoid dup from DIExpression)

    cur_df=cur_proj=cur_job=cur_schema=""

    def ctx(elem):
        # project / job / df by nearest ancestors, then global maps
        p=j=d=""
        for a in ancestors(elem, pm, 200):
            t=lower(strip_ns(getattr(a,"tag",""))); at=attrs_ci(a)
            nm=(at.get("name") or at.get("displayname") or "").strip()
            if not d and t in DF_TAGS: d=nm
            if not p and t in PROJECT_TAGS: p=nm
            if t in JOB_TAGS and not j: j=job_name_from_node(a)
        d=d or cur_df; p=p or df_proj.get(d,""); j=j or df_job.get(d,"")
        return p, j, d

    for e in root.iter():
        if not isinstance(e.tag,str): continue
        tag=lower(strip_ns(e.tag)); a=attrs_ci(e)

        # track current visibles
        if tag in PROJECT_TAGS: cur_proj=(a.get("name") or a.get("displayname") or cur_proj).strip()
        if tag in DF_TAGS:      cur_df  =(a.get("name") or a.get("displayname") or cur_df).strip()
        if tag in JOB_TAGS:     cur_job = job_name_from_node(e) or cur_job
        if tag=="dischema":     cur_schema=(a.get("name") or a.get("displayname") or cur_schema).strip()

        # ---- sources / targets
        if tag in ("didatabasetablesource","didatabasetabletarget"):
            ds =(a.get("datastorename") or a.get("datastore") or "").strip()
            sch=(a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl=(a.get("tablename") or a.get("table") or "").strip()
            if ds and tbl:
                p,j,d = ctx(e)
                role="source" if "source" in tag else "target"
                source_target.add((p,j,d,role,_norm_key(ds),_norm_key(sch),_norm_key(tbl), ds,sch,tbl))

        # ---- capture ALL FUNCTION_CALL lookup / lookup_ext
        if tag=="function_call" and lower(a.get("name","")) in ("lookup","lookup_ext"):
            is_ext = (lower(a.get("name"))=="lookup_ext")
            p,j,d = ctx(e)
            schema_out = nearest_schema_out(e, pm, cur_schema)

            # accept multiple attribute variants
            attr_text = " ".join([f'{k}="{v}"' for k,v in a.items()])
            body_text = collect_text(e)
            ds=sch=tbl=""

            if is_ext:
                # prefer named kv
                m = LOOKUP_EXT_KV.search(attr_text) or LOOKUP_EXT_KV.search(body_text)
                if m and _valid(m.group("ds"), m.group("own"), m.group("tbl")):
                    ds, sch, tbl = m.group("ds"), m.group("own"), m.group("tbl")
                if not ds:
                    ds, sch, tbl = extract_lookup_triple(attr_text+" "+body_text, True)
            else:
                # lookup: parse args/dotted from attributes+body
                ds, sch, tbl = extract_lookup_triple(attr_text+" "+body_text, False)

            if ds and tbl and schema_out:
                key = (p,j,d,_norm_key(ds),_norm_key(sch),_norm_key(tbl))
                if is_ext:
                    lookup_ext_s[key].add(schema_out)
                    seen_ext_fun.add(key)
                else:
                    col = nearest_output_col(e, pm)
                    if col:
                        lookup_pos[key].append(f"{schema_out}>>{col}")

        # ---- fallback DIExpression / DIAttribute containing lookup text
        if tag in ("diexpression","diattribute"):
            txt = " ".join([f'{k}="{v}"' for k,v in a.items()]) + " " + collect_text(e)
            if not (HAS_LOOKUP.search(txt) or HAS_LOOKUP_EXT.search(txt)): 
                continue
            p,j,d = ctx(e)
            schema_out = nearest_schema_out(e, pm, cur_schema)
            if not schema_out: 
                continue

            if HAS_LOOKUP_EXT.search(txt):
                ds, sch, tbl = extract_lookup_triple(txt, True)
                if ds and tbl:
                    key=(p,j,d,_norm_key(ds),_norm_key(sch),_norm_key(tbl))
                    if key not in seen_ext_fun:   # only if FUNCTION_CALL didn’t already capture it
                        lookup_ext_s[key].add(schema_out)

            if HAS_LOOKUP.search(txt):
                ds, sch, tbl = extract_lookup_triple(txt, False)
                if ds and tbl:
                    col = nearest_output_col(e, pm)
                    if col:
                        key=(p,j,d,_norm_key(ds),_norm_key(sch),_norm_key(tbl))
                        lookup_pos[key].append(f"{schema_out}>>{col}")

    # ----------------- build dataframe -----------------
    Row = namedtuple("Row", [
        "project_name","job_name","dataflow_name",
        "role","datastore","schema","table",
        "lookup_position","in_transf_used_count"
    ])
    rows=[]

    # lookups (column-level)
    for (p,j,d,dsN,schN,tblN), pos in lookup_pos.items():
        positions = ", ".join(sorted(dedupe([_pretty(x) for x in pos if x])))
        rows.append(Row(p or "", j or "", d or "", "lookup",
                        _pretty(dsN), _pretty(schN), _pretty(tblN),
                        positions, len([1 for _ in positions.split(",") if _.strip()])))

    # lookup_ext (schema-level)
    for (p,j,d,dsN,schN,tblN), posset in lookup_ext_s.items():
        positions = ", ".join(sorted(dedupe([_pretty(x) for x in posset if x])))
        rows.append(Row(p or "", j or "", d or "", "lookup_ext",
                        _pretty(dsN), _pretty(schN), _pretty(tblN),
                        positions, len([1 for _ in positions.split(",") if _.strip()])))

    # sources/targets
    for (p,j,d,role,dsN,schN,tblN, dsD,schD,tblD) in sorted(source_target):
        rows.append(Row(p or "", j or "", d or "", role,
                        _pretty(dsD), _pretty(schD), _pretty(tblD),
                        "", 0))

    df = pd.DataFrame(rows)
    if df.empty: return df

    # strict dedupe on normalized key and merge positions
    def k(r):
        return (r["project_name"], r["job_name"], r["dataflow_name"], r["role"],
                _norm_key(r["datastore"]), _norm_key(r["schema"]), _norm_key(r["table"]))
    df["__k__"]=df.apply(k, axis=1)
    agg=(df.groupby(["__k__","project_name","job_name","dataflow_name","role",
                     "datastore","schema","table"], dropna=False, as_index=False)
            .agg({"lookup_position": lambda s: ", ".join(sorted(dedupe([_pretty(x) for x in s if str(x).strip()]))) }))
    df=agg.drop(columns="__k__")
    df["in_transf_used_count"]=df["lookup_position"].apply(lambda x: len([1 for _ in str(x).split(",") if _.strip()]))

    # column order / final clean
    cols = ["project_name","job_name","dataflow_name","role","datastore","schema","table","lookup_position","in_transf_used_count"]
    for c in cols:
        if c not in df.columns: df[c]=""
    df = df[cols].sort_values(by=["project_name","job_name","dataflow_name","role","datastore","schema","table"]).reset_index(drop=True)
    return df

# ------------------------ main ------------------------
def main():
    # >>>>>> EDIT THESE <<<<<<
    xml_path = r"C:\path\to\your\export.xml"                      # project/job/df level export
    out_xlsx = r"C:\path\to\export_xml_lineage_final.xlsx"
    # >>>>>>>>>>>>>>>>>>>>>>>>

    df = parse_single_xml(xml_path)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print(f"Done. Rows: {len(df)}  ->  {out_xlsx}")

if __name__ == "__main__":
    main()
