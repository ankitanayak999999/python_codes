#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys
import xml.etree.ElementTree as ET
from collections import namedtuple, defaultdict
import pandas as pd

Record = namedtuple("Record", [
    "project_name","job_name","dataflow_name",
    "role","datastore","schema","table",
    "lookup_position","used_in_transform","lookup_used_count"
])

# -------------------- small utils --------------------
def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag) if isinstance(tag,str) else ""
def lower(s): return (s or "").strip().lower()
def attrs_ci(e): return {k.lower(): (v or "") for k,v in getattr(e,"attrib",{}).items()}
def build_parent_map(root): return {c:p for p in root.iter() for c in p}
def ancestors(e, pm, lim=200):
    cur=e
    for _ in range(lim):
        if cur is None: break
        yield cur; cur = pm.get(cur)
def nearest_anc(e, pm, tag_lc):
    for a in ancestors(e, pm):
        if lower(strip_ns(a.tag))==tag_lc: return a
    return None
def dedupe(seq):
    out, seen = [], set()
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out
def collect_text(node):
    buf=[]
    if hasattr(node,"attrib"):
        for _,v in node.attrib.items():
            if v: buf.append(str(v))
    if node.text: buf.append(node.text)
    for ch in list(node):
        buf.append(collect_text(ch))
        if ch.tail: buf.append(ch.tail)
    return " ".join([b for b in buf if b])

def schema_out_from_DISchema(e, pm, fallback=""):
    for a in ancestors(e, pm):
        if lower(strip_ns(a.tag))=="dischema":
            nm=(attrs_ci(a).get("name") or "").strip()
            if nm: return nm
    return fallback or ""

def find_output_column(e, pm):
    if lower(strip_ns(e.tag))=="dielement":
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

# -------------------- context (project / job / df) --------------------
DF_TAGS=("didataflow","dataflow","dflow")
JOB_TAGS=("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS=("diproject","project")
WF_TAGS=("diworkflow","workflow")
CALLSTEP_TAGS=("dicallstep","callstep")

def collect_df_names(root):
    out=set()
    for n in root.iter():
        if lower(strip_ns(getattr(n,"tag",""))) in DF_TAGS:
            nm=(n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: out.add(nm)
    return out

def _normalize_candidates(val):
    if not val: return []
    raw = val.strip().strip('"').strip("'")
    parts = {raw}
    for sep in ("/","\\",".",":"):
        if sep in raw: parts.add(raw.split(sep)[-1])
    return [p.strip() for p in parts if p.strip()]

def build_workflow_df_map(root, df_names):
    wf_df=defaultdict(set)
    for wn in root.iter():
        if lower(strip_ns(getattr(wn,"tag",""))) not in WF_TAGS: continue
        wf_name=(wn.attrib.get("name") or wn.attrib.get("displayName") or "").strip()
        if not wf_name: continue
        for call in wn.iter():
            if lower(strip_ns(getattr(call,"tag",""))) in CALLSTEP_TAGS:
                for _,v in getattr(call,"attrib",{}).items():
                    for cand in _normalize_candidates(v):
                        for df in df_names:
                            if cand.lower()==df.lower():
                                wf_df[wf_name].add(df)
    return wf_df

def _job_name_from_node(job_node):
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch,"tag","")))== "diattribute" and lower(ch.attrib.get("name",""))=="job_name":
            v=(ch.attrib.get("value") or "").strip()
            if v: return v
    return (job_node.attrib.get("name") or job_node.attrib.get("displayName") or "").strip()

def build_df_project_map(root):
    df_names=collect_df_names(root)
    df_proj={}
    for pn in root.iter():
        if lower(strip_ns(getattr(pn,"tag",""))) not in PROJECT_TAGS: continue
        proj=(pn.attrib.get("name") or pn.attrib.get("displayName") or "").strip()
        if not proj: continue
        for df_node in pn.iter():
            if lower(strip_ns(getattr(df_node,"tag",""))) in DF_TAGS:
                nm=(df_node.attrib.get("name") or df_node.attrib.get("displayName") or "").strip()
                if nm: df_proj.setdefault(nm, proj)
        for desc in pn.iter():
            for _,v in getattr(desc,"attrib",{}).items():
                for cand in _normalize_candidates(v):
                    for df in df_names:
                        if cand.lower()==df.lower():
                            df_proj.setdefault(df, proj)
    return df_proj

def build_df_job_map(root):
    df_names=collect_df_names(root)
    wf_df=build_workflow_df_map(root, df_names)
    df_job={}
    for jn in root.iter():
        if lower(strip_ns(getattr(jn,"tag",""))) not in JOB_TAGS: continue
        job=_job_name_from_node(jn)
        if not job: continue
        for df_node in jn.iter():
            if lower(strip_ns(getattr(df_node,"tag",""))) in DF_TAGS:
                nm=(df_node.attrib.get("name") or df_node.attrib.get("displayName") or "").strip()
                if nm: df_job.setdefault(nm, job)
        for call in jn.iter():
            if lower(strip_ns(getattr(call,"tag",""))) in CALLSTEP_TAGS:
                for _,v in getattr(call,"attrib",{}).items():
                    for cand in _normalize_candidates(v):
                        if cand in wf_df:
                            for df in wf_df[cand]: df_job.setdefault(df, job)
                        for df in df_names:
                            if cand.lower()==df.lower():
                                df_job.setdefault(df, job)
        for desc in jn.iter():
            for _,v in getattr(desc,"attrib",{}).items():
                for cand in _normalize_candidates(v):
                    for df in df_names:
                        if cand.lower()==df.lower():
                            df_job.setdefault(df, job)
    return df_job

# -------------------- lookup extractors --------------------
DOT_NORMALIZE = re.compile(r"\s*\.\s*")
NAME_CHARS = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"

# dot style
LOOKUP_CALL_RE      = re.compile(rf'lookup\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)
LOOKUP_EXT_CALL_RE  = re.compile(rf'lookup_ext\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)

# comma args
LOOKUP_CALL_ARGS_RE = re.compile(rf'lookup\s*\(\s*\{{?\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})', re.I|re.S)
LOOKUP_EXT_ARGS_RE  = re.compile(rf'lookup_ext\s*\(\s*\{{?\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})', re.I|re.S)

# **brace-wrapped triple** { DS, SCHEMA, TABLE }  (handled FIRST)
BRACED_TRIPLE_EXT = re.compile(
    r'lookup[_\s]*ext\s*\(\s*\{\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*,\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*,\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*\}',
    re.I|re.S
)
BRACED_TRIPLE     = re.compile(
    r'lookup\s*\(\s*\{\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*,\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*,\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*\}',
    re.I|re.S
)

# attrs blob (FUNCTION_CALL)
LOOKUP_ATTR_RE = re.compile(
    r'tabledatastore\s*=\s*(?P<q1>[\'"]|&quot;)?(?P<ds>[^\'"&\s]+).*?'
    r'tableowner\s*=\s*(?P<q2>[\'"]|&quot;)?(?P<own>[^\'"&\s]+).*?'
    r'tablename\s*=\s*(?P<q3>[\'"]|&quot;)?(?P<tbl>[^\'"&\s]+)',
    re.I | re.S
)

# named key=value inside call text (lookup_ext only)
LOOKUP_EXT_NAMED_KV_RE = re.compile(
    r'lookup_ext\s*\([^)]*?'
    r'(?:tableDatastore|tabledatastore)\s*=\s*([\'"]?)(?P<ds>[^\'",)\s]+)\1[^)]*?'
    r'(?:tableOwner|tableowner)\s*=\s*([\'"]?)(?P<own>[^\'",)\s]+)\3[^)]*?'
    r'(?:tableName|tablename)\s*=\s*([\'"]?)(?P<tbl>[^\'",)\s]+)\5',
    re.I | re.S
)

def extract_lookup_from_call(text: str, is_ext: bool = False):
    """Return (datastore, owner/schema, table) with all styles supported."""
    if not text: return ("","","")

    # 1) { DS, SCHEMA, TABLE } — do FIRST to avoid duplicates
    m0 = (BRACED_TRIPLE_EXT if is_ext else BRACED_TRIPLE).search(text)
    if m0:
        return m0.group(1).strip(), m0.group(2).strip(), m0.group(3).strip()

    # 2) DS.SCHEMA.TABLE
    tnorm = DOT_NORMALIZE.sub(".", text)
    m1 = (LOOKUP_EXT_CALL_RE if is_ext else LOOKUP_CALL_RE).search(tnorm)
    if m1:
        return m1.group(1).strip(), m1.group(2).strip(), m1.group(3).strip()

    # 3) DS, SCHEMA, TABLE
    pat = LOOKUP_EXT_ARGS_RE if is_ext else LOOKUP_CALL_ARGS_RE
    m2 = pat.search(text)
    if m2:
        return m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()

    # 4) named kv inside call (ext only)
    if is_ext:
        m3 = LOOKUP_EXT_NAMED_KV_RE.search(text)
        if m3:
            return m3.group("ds").strip(), m3.group("own").strip(), m3.group("tbl").strip()

    return ("","","")

def extract_lookup_from_attrs_blob(blob):
    m = LOOKUP_ATTR_RE.search(blob or "")
    return (m.group("ds"), m.group("own"), m.group("tbl")) if m else ("","","")

# -------------------- custom functions --------------------
UDF_TAGS = ("diuserdefinedfunction","diuserfunction","difunction","dicustomfunction","discriptfunction")
UDF_CALL_RE_TEMPLATE = r'\b({NAMES})\s*\('  # dynamic

def index_udf_lookups(root):
    """Return (udf_tables, udf_call_re). udf_tables: NAME->set((ds,sch,tbl))."""
    udf_tables = {}
    names=[]
    for n in root.iter():
        if lower(strip_ns(getattr(n,"tag",""))) in UDF_TAGS:
            name=(n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if not name: continue
            body = DOT_NORMALIZE.sub(".", collect_text(n))
            uses=set()
            for pat in (BRACED_TRIPLE_EXT, LOOKUP_EXT_CALL_RE, LOOKUP_EXT_ARGS_RE, LOOKUP_EXT_NAMED_KV_RE,
                        BRACED_TRIPLE, LOOKUP_CALL_RE, LOOKUP_CALL_ARGS_RE):
                for m in pat.finditer(body):
                    # support named-kv’s groupdict too
                    if hasattr(m, "groupdict") and "ds" in m.groupdict():
                        ds, sch, tbl = m.group("ds"), m.group("own"), m.group("tbl")
                    else:
                        ds, sch, tbl = m.group(1), m.group(2), m.group(3)
                    if ds and tbl: uses.add((ds.strip(), sch.strip(), tbl.strip()))
            if uses:
                key=name.upper()
                udf_tables[key]=uses
                names.append(re.escape(key))
    udf_call_re = re.compile(UDF_CALL_RE_TEMPLATE.replace("{NAMES}", "|".join(names)), re.I) if names else None
    return udf_tables, udf_call_re

# -------------------- main parser --------------------
def parse_single_xml(xml_path: str):
    tree = ET.parse(xml_path); root = tree.getroot()
    pm = build_parent_map(root)

    udf_tables, udf_call_re = index_udf_lookups(root)
    df_job_map  = build_df_job_map(root)
    df_proj_map = build_df_project_map(root)

    rows_src_tgt = []
    seen_src_tgt = set()

    # (proj,job,df,ds,sch,tbl) -> positions
    lookup_pos = defaultdict(list)   # for lookup (column-level)
    lookup_ext_pos = defaultdict(set) # for lookup_ext

    cur_proj=cur_job=cur_df=cur_schema=""
    last_job=""

    def context_for(e):
        proj=job=df=""
        for a in ancestors(e, pm):
            t=lower(strip_ns(a.tag)); at=attrs_ci(a)
            nm=(at.get("name") or at.get("displayname") or "").strip()
            if not df and t in DF_TAGS: df = nm or df
            if not proj and t in PROJECT_TAGS: proj = nm or proj
            if t in JOB_TAGS and not job: job = _job_name_from_node(a) or job
        df   = df or cur_df
        proj = proj or df_proj_map.get(df, cur_proj)
        job  = job or df_job_map.get(df, cur_job or last_job)
        return proj, job, df

    for e in root.iter():
        if not isinstance(e.tag, str): continue
        tag = lower(strip_ns(e.tag)); a=attrs_ci(e)

        if tag in PROJECT_TAGS: cur_proj=(a.get("name") or a.get("displayname") or cur_proj).strip()
        if tag in DF_TAGS: cur_df=(a.get("name") or a.get("displayname") or cur_df).strip()
        if tag in JOB_TAGS:
            cur_job = _job_name_from_node(e) or (a.get("name") or a.get("displayname") or cur_job).strip()
            if cur_job: last_job=cur_job
        if tag=="dischema": cur_schema=(a.get("name") or a.get("displayname") or cur_schema).strip()

        # build a blob to search
        blob_parts = [" ".join([f'{k}="{v}"' for k,v in a.items()])]
        for ch in list(e):
            t=lower(strip_ns(getattr(ch,"tag","")))
            if t in ("diattribute","diexpression","attribute","param","parameter","property","constant"):
                val = ch.attrib.get("value") or ch.attrib.get("expr") or (ch.text or "")
                if val: blob_parts.append(str(val))
        blob = " ".join(blob_parts)

        # -------- sources / targets --------
        if tag in ("didatabasetablesource","didatabasetabletarget"):
            ds=(a.get("datastorename") or "").strip()
            sch=(a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl=(a.get("tablename") or "").strip()
            if ds and tbl:
                proj,job,df=context_for(e)
                role="source" if "source" in tag else "target"
                key=(proj,job,df,role,ds,sch,tbl,"","",0)
                if key not in seen_src_tgt:
                    seen_src_tgt.add(key)
                    rows_src_tgt.append(Record(proj,job,df,role,ds,sch,tbl,"","",0))

        # -------- ui_mapping_text (lookup + UDF calls) --------
        if tag=="diattribute" and lower(a.get("name",""))=="ui_mapping_text":
            txt = a.get("value") or (e.text or "")
            proj,job,df=context_for(e)
            schema_out=schema_out_from_DISchema(e, pm, cur_schema)
            col=find_output_column(e, pm)
            ds1,sch1,tb1 = extract_lookup_from_call(txt, is_ext=False)
            if ds1 and tb1 and schema_out and col:
                lookup_pos[(proj,job,df,ds1,sch1,tb1)].append(f"{schema_out}>>{col}")
            if udf_call_re and udf_call_re.search(txt or ""):
                pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out
                if pos:
                    for m in udf_call_re.finditer(txt or ""):
                        udf = m.group(1).upper()
                        for (dsu,schu,tblu) in udf_tables.get(udf, set()):
                            lookup_ext_pos[(proj,job,df,dsu,schu,tblu)].add(pos)

        # -------- DIExpression / DIAttribute expr/value --------
        if tag in ("diexpression","diattribute"):
            expr = a.get("expr") or a.get("value") or (e.text or "")
            if expr:
                proj,job,df=context_for(e)
                schema_out=schema_out_from_DISchema(e, pm, cur_schema)
                col=find_output_column(e, pm)
                dsx,schx,tbx = extract_lookup_from_call(expr, is_ext=True)
                if dsx and tbx:
                    pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out
                    if pos: lookup_ext_pos[(proj,job,df,dsx,schx,tbx)].add(pos)
                dsl,schl,tbl = extract_lookup_from_call(expr, is_ext=False)
                if dsl and tbl and schema_out and col:
                    lookup_pos[(proj,job,df,dsl,schl,tbl)].append(f"{schema_out}>>{col}")
                if udf_call_re and udf_call_re.search(expr):
                    pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out
                    if pos:
                        for m in udf_call_re.finditer(expr):
                            udf = m.group(1).upper()
                            for (dsu,schu,tblu) in udf_tables.get(udf, set()):
                                lookup_ext_pos[(proj,job,df,dsu,schu,tblu)].add(pos)

        # -------- FUNCTION_CALL (attrs / kv / dot / comma + UDF) --------
        if tag=="function_call":
            fn = lower(a.get("name",""))
            proj,job,df=context_for(e)
            schema_out=schema_out_from_DISchema(e, pm, cur_schema)
            col=find_output_column(e, pm)
            # attributes
            ds=(a.get("tabledatastore") or a.get("datastorename") or "").strip()
            sch=(a.get("tableowner") or a.get("ownername") or "").strip()
            tbl=(a.get("tablename") or "").strip()
            blob_all = collect_text(e)
            if not (ds and tbl) and ("tabledatastore" in lower(blob_all) and "tablename" in lower(blob_all)):
                ds,sch,tbl = extract_lookup_from_attrs_blob(blob_all)
            if not (ds and tbl):
                ds2,sch2,tbl2 = extract_lookup_from_call(blob_all, is_ext=("lookup_ext" in fn))
                ds,sch,tbl = (ds2 or ds), (sch2 or sch), (tbl2 or tbl)
            if ds and tbl:
                pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out
                if "lookup_ext" in fn:
                    if pos: lookup_ext_pos[(proj,job,df,ds,sch,tbl)].add(pos)
                elif "lookup" in fn:
                    if schema_out and col:
                        lookup_pos[(proj,job,df,ds,sch,tbl)].append(f"{schema_out}>>{col}")
            # UDF as function name
            fn_name = (a.get("name") or "").strip().upper()
            if fn_name in udf_tables:
                pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out
                if pos:
                    for (dsu,schu,tblu) in udf_tables[fn_name]:
                        lookup_ext_pos[(proj,job,df,dsu,schu,tblu)].add(pos)

        # -------- generic fallback on any node --------
        blob_any = collect_text(e)
        if "lookup_ext(" in lower(blob_any) or "lookup(" in lower(blob_any) or (udf_call_re and udf_call_re.search(blob_any)):
            proj,job,df=context_for(e)
            schema_out=schema_out_from_DISchema(e, pm, cur_schema)
            col=find_output_column(e, pm)
            pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out
            dsx,schx,tbx = extract_lookup_from_call(blob_any, is_ext=True)
            if dsx and tbx and pos:
                lookup_ext_pos[(proj,job,df,dsx,schx,tbx)].add(pos)
            dsl,schl,tbl = extract_lookup_from_call(blob_any, is_ext=False)
            if dsl and tbl and schema_out and col:
                lookup_pos[(proj,job,df,dsl,schl,tbl)].append(f"{schema_out}>>{col}")
            if udf_call_re:
                for m in udf_call_re.finditer(blob_any):
                    udf = m.group(1).upper()
                    for (dsu,schu,tblu) in udf_tables.get(udf, set()):
                        if pos: lookup_ext_pos[(proj,job,df,dsu,schu,tblu)].add(pos)

    # -------------------- emit rows --------------------
    out=[]
    # lookup (column-level)
    for (proj,job,df,ds,sch,tbl), pos_list in lookup_pos.items():
        uniq = dedupe(pos_list)
        if uniq:
            out.append(Record(proj,job,df,"lookup",ds,sch,tbl,", ".join(uniq),"lookup",len(uniq)))
    # lookup_ext
    for (proj,job,df,ds,sch,tbl), pos_set in lookup_ext_pos.items():
        uniq = dedupe(list(pos_set))
        out.append(Record(proj,job,df,"lookup_ext",ds,sch,tbl,", ".join(uniq),"lookup_ext",len(uniq)))
    # sources/targets
    out.extend(rows_src_tgt)

    out.sort(key=lambda r:(r.project_name or "", r.job_name or "", r.dataflow_name or "",
                           r.role, r.datastore or "", r.schema or "", r.table or "", r.lookup_position or ""))
    return out

# -------------------- run --------------------
def main():
    # <<< SET THIS PATH >>>
    xml_path = r"C:\path\to\export.xml"
    # <<< --------------- >>>

    if not os.path.isfile(xml_path):
        print(f"File not found: {xml_path}"); sys.exit(1)

    rows = parse_single_xml(xml_path)
    df = pd.DataFrame([r._asdict() for r in rows]) if rows else pd.DataFrame(columns=Record._fields)

    out_base = os.path.splitext(xml_path)[0] + "_lineage_final"
    df.to_csv(out_base + ".csv", index=False)
    with pd.ExcelWriter(out_base + ".xlsx") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print("Done.")
    print(f"CSV : {out_base}.csv")
    print(f"XLSX: {out_base}.xlsx")

if __name__ == "__main__":
    main()
