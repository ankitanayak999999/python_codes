#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import namedtuple, defaultdict
import pandas as pd

# ============================ Output Row ============================
Record = namedtuple("Record", [
    "project_name",
    "job_name",
    "dataflow_name",
    "role",                 # source / target / lookup / lookup_ext
    "datastore",
    "schema",
    "table",
    "lookup_position",      # lookup: "SchemaOut>>Col, ..."; lookup_ext: "SchemaOut>>Col" or "SchemaOut"; src/tgt: ""
    "used_in_transform",    # "lookup" / "lookup_ext" (blank for src/tgt)
    "lookup_used_count"     # unique positions count
])

# ============================ Small Utils ============================
def strip_ns(tag: str) -> str:
    return re.sub(r"^\{.*\}", "", tag).strip() if isinstance(tag, str) else ""

def lower(s: str) -> str:
    return (s or "").strip().lower()

def attrs_ci(elem) -> dict:
    return {k.lower(): (v or "") for k, v in getattr(elem, "attrib", {}).items()}

def build_parent_map(root):
    return {c: p for p in root.iter() for c in p}

def ancestors(elem, parent_map, max_up: int = 200):
    cur = elem
    for _ in range(max_up):
        if cur is None:
            break
        yield cur
        cur = parent_map.get(cur)

def nearest_ancestor(elem, parent_map, tag_name_lc: str):
    for anc in ancestors(elem, parent_map):
        if lower(strip_ns(anc.tag)) == tag_name_lc:
            return anc
    return None

def dedupe_preserve_order(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def collect_text(node) -> str:
    buf = []
    if hasattr(node, "attrib"):
        for _, v in node.attrib.items():
            if v:
                buf.append(str(v))
    if node.text:
        buf.append(node.text or "")
    for ch in list(node):
        buf.append(collect_text(ch))
        if ch.tail:
            buf.append(ch.tail or "")
    return " ".join([b for b in buf if b])

# ============================ Column/Schema Helpers ============================
def schema_out_from_DISchema(elem, parent_map, fallback="") -> str:
    for anc in ancestors(elem, parent_map):
        if lower(strip_ns(anc.tag)) == "dischema":
            nm = (attrs_ci(anc).get("name") or "").strip()
            if nm:
                return nm
    return fallback or ""

def find_output_column(elem, parent_map) -> str:
    if lower(strip_ns(elem.tag)) == "dielement":
        nm = (attrs_ci(elem).get("name") or "").strip()
        if nm:
            return nm
    cur = elem
    for _ in range(200):
        if cur is None:
            break
        if lower(strip_ns(cur.tag)) == "dielement":
            nm = (attrs_ci(cur).get("name") or "").strip()
            if nm:
                return nm
        cur = parent_map.get(cur)
    return ""

# ============================ Context (Project/Workflow/Job/DF) ============================
DF_TAGS        = ("didataflow", "dataflow", "dflow")
JOB_TAGS       = ("dijob", "dibatchjob", "job", "batch_job")
PROJECT_TAGS   = ("diproject", "project")
WORKFLOW_TAGS  = ("diworkflow", "workflow")
CALLSTEP_TAGS  = ("dicallstep", "callstep")

def collect_df_names(root):
    names = set()
    for n in root.iter():
        if lower(strip_ns(getattr(n, "tag", ""))) in DF_TAGS:
            nm = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm:
                names.add(nm)
    return names

def _normalize_candidates(val: str):
    if not val:
        return []
    raw = val.strip().strip('"').strip("'")
    parts = {raw}
    for sep in ("/", "\\", ".", ":"):
        if sep in raw:
            parts.add(raw.split(sep)[-1])
    return [p.strip() for p in parts if p.strip()]

def build_workflow_df_map(root, df_names):
    wf_df = defaultdict(set)
    for wn in root.iter():
        if lower(strip_ns(getattr(wn, "tag", ""))) not in WORKFLOW_TAGS:
            continue
        wf_name = (wn.attrib.get("name") or wn.attrib.get("displayName") or "").strip()
        if not wf_name:
            continue
        for call in wn.iter():
            if lower(strip_ns(getattr(call, "tag", ""))) in CALLSTEP_TAGS:
                for _, v in getattr(call, "attrib", {}).items():
                    for cand in _normalize_candidates(v):
                        for df in df_names:
                            if cand.lower() == df.lower():
                                wf_df[wf_name].add(df)
    return wf_df

def build_df_project_map(root):
    df_names = collect_df_names(root)
    df_proj = {}
    for pn in root.iter():
        if lower(strip_ns(getattr(pn, "tag", ""))) not in PROJECT_TAGS:
            continue
        proj = (pn.attrib.get("name") or pn.attrib.get("displayName") or "").strip()
        if not proj:
            continue
        for df_node in pn.iter():
            if lower(strip_ns(getattr(df_node, "tag", ""))) in DF_TAGS:
                nm = (df_node.attrib.get("name") or df_node.attrib.get("displayName") or "").strip()
                if nm:
                    df_proj.setdefault(nm, proj)
        for desc in pn.iter():
            if not hasattr(desc, "attrib"):
                continue
            for _, v in desc.attrib.items():
                for cand in _normalize_candidates(v):
                    for df in df_names:
                        if cand.lower() == df.lower():
                            df_proj.setdefault(df, proj)
    return df_proj

def _job_name_from_node(job_node) -> str:
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch, "tag", ""))) == "diattribute" and lower(ch.attrib.get("name", "")) == "job_name":
            v = (ch.attrib.get("value") or "").strip()
            if v:
                return v
    return (job_node.attrib.get("name") or job_node.attrib.get("displayName") or "").strip()

def build_df_job_map(root):
    df_names = collect_df_names(root)
    wf_df = build_workflow_df_map(root, df_names)
    df_job = {}
    for jn in root.iter():
        if lower(strip_ns(getattr(jn, "tag", ""))) not in JOB_TAGS:
            continue
        job_name = _job_name_from_node(jn)
        if not job_name:
            continue
        for df_node in jn.iter():
            if lower(strip_ns(getattr(df_node, "tag", ""))) in DF_TAGS:
                nm = (df_node.attrib.get("name") or df_node.attrib.get("displayName") or "").strip()
                if nm:
                    df_job.setdefault(nm, job_name)
        for call in jn.iter():
            if lower(strip_ns(getattr(call, "tag", ""))) in CALLSTEP_TAGS:
                for _, v in getattr(call, "attrib", {}).items():
                    for cand in _normalize_candidates(v):
                        if cand in wf_df:
                            for df in wf_df[cand]:
                                df_job.setdefault(df, job_name)
                        for df in df_names:
                            if cand.lower() == df.lower():
                                df_job.setdefault(df, job_name)
        for desc in jn.iter():
            if not hasattr(desc, "attrib"):
                continue
            for _, v in desc.attrib.items():
                for cand in _normalize_candidates(v):
                    for df in df_names:
                        if cand.lower() == df.lower():
                            df_job.setdefault(df, job_name)
    return df_job

# ============================ Lookup Patterns/Extractors ============================
DOT_NORMALIZE = re.compile(r"\s*\.\s*")
# Expanded char set to include [] @ %
NAME_CHARS    = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"

# dot-style DS.SCHEMA.TABLE
LOOKUP_CALL_RE     = re.compile(
    rf'lookup\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?',
    re.IGNORECASE
)
LOOKUP_EXT_CALL_RE = re.compile(
    rf'lookup_ext\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?',
    re.IGNORECASE
)
# comma-arg style DS, SCHEMA, TABLE (allow optional { } and quotes and newlines)
LOOKUP_CALL_ARGS_RE = re.compile(
    rf'lookup\s*\(\s*\{{?\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?',
    re.IGNORECASE | re.DOTALL
)
LOOKUP_EXT_CALL_ARGS_RE = re.compile(
    rf'lookup_ext\s*\(\s*\{{?\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?',
    re.IGNORECASE | re.DOTALL
)
# NEW: named key=value style inside the function text (order can vary, tolerate quotes/newlines)
LOOKUP_EXT_NAMED_KV_RE = re.compile(
    r'lookup_ext\s*\([^)]*?'
    r'(?:tableDatastore|tabledatastore)\s*=\s*([\'"]?)(?P<ds>[^\'",)\s]+)\1[^)]*?'
    r'(?:tableOwner|tableowner)\s*=\s*([\'"]?)(?P<own>[^\'",)\s]+)\3[^)]*?'
    r'(?:tableName|tablename)\s*=\s*([\'"]?)(?P<tbl>[^\'",)\s]+)\5'
    r'[^)]*\)',
    re.IGNORECASE | re.DOTALL
)

def extract_lookup_from_call(text: str, is_ext: bool = False):
    """Return (datastore, owner/schema, table) from dot / comma-arg / (for ext) named-kv styles."""
    if not text:
        return ("", "", "")
    # 1) dot-style
    tnorm = DOT_NORMALIZE.sub(".", text)
    m = (LOOKUP_EXT_CALL_RE if is_ext else LOOKUP_CALL_RE).search(tnorm)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    # 2) comma-arg style
    pat = LOOKUP_EXT_CALL_ARGS_RE if is_ext else LOOKUP_CALL_ARGS_RE
    m2 = pat.search(text)
    if m2:
        return m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()
    # 3) (ext only) named kv inside call text
    if is_ext:
        m3 = LOOKUP_EXT_NAMED_KV_RE.search(text)
        if m3:
            return m3.group("ds").strip(), m3.group("own").strip(), m3.group("tbl").strip()
    return ("", "", "")

# FUNCTION_CALL attribute blob (e.g., tableDatastore=".." tableOwner=".." tableName="..")
LOOKUP_ATTR_RE = re.compile(
    r'tabledatastore\s*=\s*(?P<q1>[\'"]|&quot;)?(?P<ds>[^\'"&\s]+).*?'
    r'tableowner\s*=\s*(?P<q2>[\'"]|&quot;)?(?P<own>[^\'"&\s]+).*?'
    r'tablename\s*=\s*(?P<q3>[\'"]|&quot;)?(?P<tbl>[^\'"&\s]+)',
    re.IGNORECASE | re.DOTALL
)

def extract_lookup_from_attrs_blob(blob: str):
    m = LOOKUP_ATTR_RE.search(blob or "")
    return (m.group("ds"), m.group("own"), m.group("tbl")) if m else ("", "", "")

# UDFs containing lookups
UDF_NAME_RE = re.compile(r'^\s*([A-Za-z0-9_ ]+)\s*\(', re.IGNORECASE)

def build_udf_lookup_index(root):
    udf_index = {}
    for n in root.iter():
        tag = lower(strip_ns(getattr(n, "tag", "")))
        if tag in ("diuserdefinedfunction", "diuserfunction", "difunction", "dicustomfunction"):
            name = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if not name:
                continue
            body = DOT_NORMALIZE.sub(".", collect_text(n))
            uses = set()
            for pat in (LOOKUP_CALL_RE, LOOKUP_EXT_CALL_RE, LOOKUP_CALL_ARGS_RE, LOOKUP_EXT_CALL_ARGS_RE):
                for m in pat.finditer(body):
                    ds, sch, tbl = m.group(1), m.group(2), m.group(3)
                    if ds and tbl:
                        uses.add((ds, sch, tbl))
            if uses:
                udf_index[name.upper()] = uses
    return udf_index

# ============================ Core Parser ============================
def parse_single_xml(xml_path: str):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] XML parse failed: {e}")
        return []

    parent_map   = build_parent_map(root)
    udf_tables   = build_udf_lookup_index(root)
    df_job_map   = build_df_job_map(root)
    df_proj_map  = build_df_project_map(root)

    rows = []  # sources/targets
    seen_src_tgt = set()

    lookup_pos_entries   = defaultdict(list)  # (proj,job,df,ds,sch,tbl) -> ["SchemaOut>>Col", ...]
    lookup_ext_positions = defaultdict(set)   # (proj,job,df,ds,sch,tbl) -> {"SchemaOut", "SchemaOut>>Col", ...}

    # streaming context & fallback
    current_project = ""
    current_job     = ""
    current_dflow   = ""
    current_schema  = ""
    last_seen_job   = ""
    pending_ext_by_step = {}  # id(step) -> context dict

    def context_for(elem):
        proj = job = df = ""
        for anc in ancestors(elem, parent_map):
            t = lower(strip_ns(anc.tag))
            a = attrs_ci(anc)
            nm = (a.get("name") or a.get("displayname") or "").strip()
            if not df and t in DF_TAGS:
                df = nm or df
            if not proj and t in PROJECT_TAGS:
                proj = nm or proj
            if t in JOB_TAGS and not job:
                job = _job_name_from_node(anc) or job
        df   = df   or current_dflow
        proj = proj or df_proj_map.get(df, current_project)
        job  = job  or df_job_map.get(df, current_job or last_seen_job)
        return proj, job, df

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue

        tag_l = lower(strip_ns(elem.tag))
        a = attrs_ci(elem)

        # maintain streaming context
        if tag_l in PROJECT_TAGS:
            current_project = (a.get("name") or a.get("displayname") or current_project).strip()
        if tag_l in DF_TAGS:
            current_dflow = (a.get("name") or a.get("displayname") or current_dflow).strip()
        if tag_l in JOB_TAGS:
            current_job = _job_name_from_node(elem) or (a.get("name") or a.get("displayname") or current_job).strip()
            if current_job:
                last_seen_job = current_job
        if tag_l == "dischema":
            current_schema = (a.get("name") or a.get("displayname") or current_schema).strip()

        # collect blob text (for FUNCTION_CALL + generic fallback)
        blob_parts = [" ".join([f'{k}="{v}"' for k, v in a.items()])]
        for ch in list(elem):
            t = lower(strip_ns(getattr(ch, "tag", "")))
            if t in ("diattribute", "attribute", "attr", "property", "prop", "param", "parameter", "constant", "diexpression"):
                val = ch.attrib.get("value") or ch.attrib.get("expr") or (ch.text or "")
                if val:
                    blob_parts.append(str(val))
        blob = " ".join(blob_parts)

        # -------- Sources / Targets --------
        if tag_l in ("didatabasetablesource", "didatabasetabletarget"):
            ds  = (a.get("datastorename") or "").strip()
            sch = (a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl = (a.get("tablename") or "").strip()
            if ds and tbl:
                proj0, job0, df0 = context_for(elem)
                role = "source" if "source" in tag_l else "target"
                key = (proj0, job0, df0, role, ds, sch, tbl, "", "", 0)
                if key not in seen_src_tgt:
                    seen_src_tgt.add(key)
                    rows.append(Record(proj0, job0, df0, role, ds, sch, tbl, "", "", 0))

        # -------- Column-level lookup via ui_mapping_text --------
        if tag_l == "diattribute" and lower(a.get("name", "")) == "ui_mapping_text":
            txt = a.get("value") or (elem.text or "")
            if "lookup(" in lower(txt):
                ds2, sch2, tbl2 = extract_lookup_from_call(txt, is_ext=False)
                if ds2 and tbl2:
                    proj0, job0, df0 = context_for(elem)
                    col = find_output_column(elem, parent_map)
                    schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                    if schema_out and col:
                        lookup_pos_entries[(proj0, job0, df0, ds2, sch2, tbl2)].append(f"{schema_out}>>{col}")
            if txt:
                m_udf = UDF_NAME_RE.search(txt)
                if m_udf:
                    udf_name = m_udf.group(1).strip().upper()
                    if udf_name in udf_tables:
                        proj0, job0, df0 = context_for(elem)
                        schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                        col = find_output_column(elem, parent_map)
                        pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out
                        for (dsu, schu, tblu) in udf_tables[udf_name]:
                            key = (proj0, job0, df0, dsu, schu, tblu)
                            _ = lookup_ext_positions[key]
                            if pos:
                                lookup_ext_positions[key].add(pos)

        # -------- DIExpression / DIAttribute with expr/value --------
        if tag_l in ("diexpression", "diattribute"):
            expr_txt = a.get("expr") or a.get("value") or (elem.text or "")
            expr_l = lower(expr_txt)

            if "lookup_ext(" in expr_l:
                step = nearest_ancestor(elem, parent_map, "diassignmentstep")
                if step is not None:
                    proj0, job0, df0 = context_for(elem)
                    schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                    pending_ext_by_step[id(step)] = {
                        "schema_out": schema_out, "project": proj0, "job": job0, "df": df0
                    }
                dsx, schx, tblx = extract_lookup_from_call(expr_txt, is_ext=True)
                if dsx and tblx:
                    proj0, job0, df0 = context_for(elem)
                    schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                    col = find_output_column(elem, parent_map)
                    pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out
                    key = (proj0, job0, df0, dsx, schx, tblx)
                    _ = lookup_ext_positions[key]
                    if pos:
                        lookup_ext_positions[key].add(pos)

            if "lookup(" in expr_l:
                dsl, schl, tbll = extract_lookup_from_call(expr_txt, is_ext=False)
                if dsl and tbll:
                    proj0, job0, df0 = context_for(elem)
                    schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                    col = find_output_column(elem, parent_map)
                    if schema_out and col:
                        lookup_pos_entries[(proj0, job0, df0, dsl, schl, tbll)].append(f"{schema_out}>>{col}")

        # -------- FUNCTION_CALL (lookup / lookup_ext) --------
        if tag_l == "function_call":
            fn_name = lower(a.get("name", ""))

            ds  = (a.get("tabledatastore") or a.get("datastorename") or "").strip()
            sch = (a.get("tableowner")    or a.get("ownername")     or "").strip()
            tbl = (a.get("tablename")     or "").strip()

            if not (ds and tbl) and ("tabledatastore" in lower(blob) and "tablename" in lower(blob)):
                ds, sch, tbl = extract_lookup_from_attrs_blob(blob)

            if not (ds and tbl):
                # includes named-kv inside the function text for lookup_ext
                ds_c, sch_c, tbl_c = extract_lookup_from_call(blob, is_ext=("lookup_ext" in fn_name))
                ds, sch, tbl = (ds_c or ds), (sch_c or sch), (tbl_c or tbl)

            if ds and tbl:
                proj0, job0, df0 = context_for(elem)
                schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                col = find_output_column(elem, parent_map)
                pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out

                step = nearest_ancestor(elem, parent_map, "diassignmentstep")
                if step is not None and id(step) in pending_ext_by_step:
                    info = pending_ext_by_step[id(step)]
                    schema_out = info.get("schema_out") or schema_out
                    proj0  = info.get("project") or proj0
                    job0   = info.get("job") or job0
                    df0    = info.get("df") or df0
                    col = find_output_column(elem, parent_map) or col
                    pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out

                base_key = (proj0, job0, df0, ds, sch, tbl)
                if "lookup_ext" in fn_name:
                    _ = lookup_ext_positions[base_key]
                    if pos:
                        lookup_ext_positions[base_key].add(pos)
                elif "lookup" in fn_name:
                    if schema_out and col:
                        lookup_pos_entries[base_key].append(f"{schema_out}>>{col}")

        # -------- GENERIC FALLBACK (catch any missed lookup/lookup_ext) --------
        if ("lookup_ext(" in lower(blob)) or ("lookup(" in lower(blob)):
            proj0, job0, df0 = context_for(elem)
            schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
            col = find_output_column(elem, parent_map)
            pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out

            # lookup_ext (try dot/comma/named-kv)
            dsx, schx, tblx = extract_lookup_from_call(blob, is_ext=True)
            if dsx and tblx:
                key = (proj0, job0, df0, dsx, schx, tblx)
                _ = lookup_ext_positions[key]
                if pos:
                    lookup_ext_positions[key].add(pos)

            # plain lookup (column-level)
            dsl, schl, tbll = extract_lookup_from_call(blob, is_ext=False)
            if dsl and tbll and schema_out and col:
                lookup_pos_entries[(proj0, job0, df0, dsl, schl, tbll)].append(f"{schema_out}>>{col}")

    # ============================ Emit Rows ============================
    out_rows = []

    for (proj, job, df, ds, sch, tbl), entries in lookup_pos_entries.items():
        uniq = dedupe_preserve_order(entries)
        if uniq:
            out_rows.append(Record(proj, job, df, "lookup", ds, sch, tbl, ", ".join(uniq), "lookup", len(uniq)))

    for (proj, job, df, ds, sch, tbl), posset in lookup_ext_positions.items():
        uniq = dedupe_preserve_order(list(posset))
        out_rows.append(Record(proj, job, df, "lookup_ext", ds, sch, tbl, ", ".join(uniq), "lookup_ext", len(uniq)))

    # include the sources/targets gathered earlier
    out_rows.extend(rows)

    out_rows.sort(key=lambda r: (
        r.project_name or "", r.job_name or "", r.dataflow_name or "",
        r.role, r.datastore or "", r.schema or "", r.table or "", r.lookup_position or ""
    ))
    return out_rows

# ============================ Main ============================
def main():
    # <<< CHANGE THIS ONLY >>>
    xml_path = r"C:\path\to\your\export.xml"
    # <<< -------------------- >>>

    if not os.path.isfile(xml_path):
        print(f"File not found: {xml_path}")
        sys.exit(1)

    rows = parse_single_xml(xml_path)
    df = pd.DataFrame([r._asdict() for r in rows]) if rows else pd.DataFrame(columns=Record._fields)

    out_base = os.path.splitext(xml_path)[0] + "_lineage_v4b"
    df.to_csv(out_base + ".csv", index=False)
    with pd.ExcelWriter(out_base + ".xlsx") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print("Done.")
    print(f"CSV : {out_base}.csv")
    print(f"XLSX: {out_base}.xlsx")

if __name__ == "__main__":
    main()
