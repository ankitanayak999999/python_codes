#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, xml.etree.ElementTree as ET
from collections import namedtuple, defaultdict
import pandas as pd

Record = namedtuple("Record", [
    "project_name",
    "job_name",
    "dataflow_name",
    "role",                 # source / target / lookup / lookup_ext
    "datastore",
    "schema",
    "table",
    "lookup_position",      # lookup: "Schema>>Col, ..."; lookup_ext: "Schema>>Col" or "Schema"; src/tgt: ""
    "used_in_transform",    # "lookup" / "lookup_ext" (blank for src/tgt)
    "lookup_used_count"     # UNIQUE positions count
])

# ---------------- helpers ----------------
def strip_ns(tag): 
    return re.sub(r"^\{.*\}", "", tag).strip() if isinstance(tag, str) else ""

def lower(s): 
    return (s or "").strip().lower()

def build_parent_map(root): 
    return {c: p for p in root.iter() for c in p}

def attrs_ci(elem): 
    return {k.lower(): (v or "") for k, v in getattr(elem, "attrib", {}).items()}

def ancestors(elem, parent_map, max_up=200):
    cur = elem
    for _ in range(max_up):
        if cur is None: break
        yield cur
        cur = parent_map.get(cur)

def nearest_ancestor(elem, parent_map, tag_name_lc):
    for anc in ancestors(elem, parent_map):
        if lower(strip_ns(anc.tag)) == tag_name_lc:
            return anc
    return None

def dedupe_preserve_order(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def find_context(elem, parent_map):
    """Return (project, job, dataflow) looking up the ancestor chain."""
    job = dflow = proj = ""
    for anc in ancestors(elem, parent_map):
        t = lower(strip_ns(anc.tag))
        a = attrs_ci(anc)
        nm = (a.get("name") or a.get("displayname") or "").strip()
        if not dflow and t in ("didataflow","dataflow","dflow"):
            dflow = nm or dflow
        if not job and t in ("dibatchjob","dijob","diworkflow","batch_job","job","workflow"):
            job = nm or job
        if not proj and t in ("diproject","project"):
            proj = nm or proj
    return proj, job, dflow

def schema_out_from_DISchema(elem, parent_map, fallback=""):
    for anc in ancestors(elem, parent_map):
        if lower(strip_ns(anc.tag)) == "dischema":
            nm = (attrs_ci(anc).get("name") or "").strip()
            if nm: return nm
    return fallback or ""

def find_output_column(elem, parent_map):
    if lower(strip_ns(elem.tag)) == "dielement":
        nm = (attrs_ci(elem).get("name") or "").strip()
        if nm: return nm
    for anc in ancestors(elem, parent_map):
        if lower(strip_ns(anc.tag)) == "dielement":
            nm = (attrs_ci(anc).get("name") or "").strip()
            if nm: return nm
    return ""

def collect_text(node):
    buf = []
    if hasattr(node, "attrib"):
        for _, v in node.attrib.items():
            if v: buf.append(str(v))
    if node.text: buf.append(node.text or "")
    for ch in list(node):
        buf.append(collect_text(ch))
        if ch.tail: buf.append(ch.tail or "")
    return " ".join([b for b in buf if b])

# ---------------- lookup regexes ----------------
DOT_NORMALIZE = re.compile(r"\s*\.\s*")
LOOKUP_CALL_RE     = re.compile(
    r'lookup\s*\(\s*"?\s*([A-Za-z0-9_]+)\s*"?\.\s*"?\s*([A-Za-z0-9_]+)\s*"?\.\s*"?\s*([A-Za-z0-9_]+)\s*"?',
    re.IGNORECASE
)
LOOKUP_EXT_CALL_RE = re.compile(
    r'lookup_ext\s*\(\s*"?\s*([A-Za-z0-9_]+)\s*"?\.\s*"?\s*([A-Za-z0-9_]+)\s*"?\.\s*"?\s*([A-Za-z0-9_]+)\s*"?',
    re.IGNORECASE
)

def extract_lookup_from_call(text, is_ext=False):
    if not text: return ("","","")
    text = DOT_NORMALIZE.sub(".", text)
    m = (LOOKUP_EXT_CALL_RE if is_ext else LOOKUP_CALL_RE).search(text)
    if not m: return ("","","")
    return m.group(1), m.group(2), m.group(3)

LOOKUP_ATTR_RE = re.compile(
    r'tabledatastore\s*=\s*(?P<q1>[\'"]|&quot;)?(?P<ds>[^\'"&\s]+).*?'
    r'tableowner\s*=\s*(?P<q2>[\'"]|&quot;)?(?P<own>[^\'"&\s]+).*?'
    r'tablename\s*=\s*(?P<q3>[\'"]|&quot;)?(?P<tbl>[^\'"&\s]+)',
    re.IGNORECASE | re.DOTALL
)
def extract_lookup_from_attrs_blob(blob: str):
    m = LOOKUP_ATTR_RE.search(blob or "")
    return (m.group("ds"), m.group("own"), m.group("tbl")) if m else ("","","")

# --------- UDF indexing: UDF -> set of (ds, sch, tbl) lookups ----------
UDF_NAME_RE = re.compile(r'^\s*([A-Za-z0-9_ ]+)\s*\(', re.IGNORECASE)

def build_udf_lookup_index(root):
    udf_index = {}
    for n in root.iter():
        tag = lower(strip_ns(getattr(n, "tag", "")))
        if tag in ("diuserdefinedfunction","diuserfunction","difunction","dicustomfunction"):
            name = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if not name: continue
            body = DOT_NORMALIZE.sub(".", collect_text(n))
            uses = set()
            for pat in (LOOKUP_CALL_RE, LOOKUP_EXT_CALL_RE):
                for m in pat.finditer(body):
                    ds, sch, tbl = m.group(1), m.group(2), m.group(3)
                    if ds and tbl: uses.add((ds, sch, tbl))
            if uses:
                udf_index[name.upper()] = uses
    return udf_index

# --------- Mappers for Job/Project <-> DataFlow ----------
JOB_TAGS     = ("dibatchjob","diworkflow","batch_job","workflow","job")
DF_TAGS      = ("didataflow","dataflow","dflow")
PROJECT_TAGS = ("diproject","project")

def normalize_candidates(val: str):
    if not val: return []
    raw = val.strip().strip('"').strip("'")
    parts = {raw}
    for sep in ("/", "\\", ".", ":"):
        if sep in raw:
            parts.add(raw.split(sep)[-1])
    return [p.strip() for p in parts if p.strip()]

def collect_df_names(root):
    names = set()
    for n in root.iter():
        if lower(strip_ns(getattr(n, "tag", ""))) in DF_TAGS:
            nm = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: names.add(nm)
    return names

def build_df_job_map(root):
    df_names = collect_df_names(root)
    df_job = {}
    for jn in root.iter():
        if lower(strip_ns(getattr(jn, "tag", ""))) not in JOB_TAGS:
            continue
        job_name = (jn.attrib.get("name") or jn.attrib.get("displayName") or "").strip()
        if not job_name: 
            continue
        for desc in jn.iter():
            if not hasattr(desc, "attrib"): 
                continue
            for _, v in desc.attrib.items():
                for cand in normalize_candidates(v):
                    for df in df_names:
                        if cand.lower() == df.lower():
                            df_job.setdefault(df, job_name)
    return df_job

def build_df_project_map(root):
    df_names = collect_df_names(root)
    df_proj = {}
    for pn in root.iter():
        if lower(strip_ns(getattr(pn, "tag", ""))) not in PROJECT_TAGS:
            continue
        proj_name = (pn.attrib.get("name") or pn.attrib.get("displayName") or "").strip()
        if not proj_name: 
            continue
        # references inside subtree
        for desc in pn.iter():
            if not hasattr(desc, "attrib"): 
                continue
            for _, v in desc.attrib.items():
                for cand in normalize_candidates(v):
                    for df in df_names:
                        if cand.lower() == df.lower():
                            df_proj.setdefault(df, proj_name)
        # nested dataflows directly
        for df_node in pn.iter():
            if lower(strip_ns(getattr(df_node, "tag",""))) in DF_TAGS:
                nm = (df_node.attrib.get("name") or df_node.attrib.get("displayName") or "").strip()
                if nm:
                    df_proj.setdefault(nm, proj_name)
    return df_proj

# --------------- core ---------------
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

    rows, seen = [], set()

    # Collect positions
    lookup_pos_entries   = defaultdict(list)  # (proj,job,df,ds,sch,tbl) -> ["Schema>>Col", ...]
    lookup_ext_positions = defaultdict(set)   # (proj,job,df,ds,sch,tbl) -> {"Schema", "Schema>>Col", ...}

    # streaming context + linkage for assignment steps
    current_project = ""
    current_job     = ""
    current_dflow   = ""
    current_schema  = ""
    pending_ext_by_step = {}  # id(<DIAssignmentStep>) -> {"schema_out","project","job","dflow"}

    for elem in root.iter():
        if not isinstance(elem.tag, str): 
            continue

        tag_l = lower(strip_ns(elem.tag))
        a = attrs_ci(elem)

        # collect any expression/attribute text under this node
        blob_parts = [" ".join([f'{k}="{v}"' for k, v in a.items()])]
        for ch in list(elem):
            t = lower(strip_ns(getattr(ch, "tag", "")))
            if t in ("diattribute","attribute","attr","property","prop","param","parameter","constant","diexpression"):
                val = ch.attrib.get("value") or ch.attrib.get("expr") or (ch.text or "")
                if val: blob_parts.append(str(val))
        blob = " ".join(blob_parts)

        # streaming context
        if tag_l in ("diproject","project"):
            current_project = (a.get("name") or a.get("displayname") or current_project).strip()
            continue
        if tag_l in ("didataflow","dataflow","dflow"):
            current_dflow = (a.get("name") or a.get("displayname") or current_dflow).strip()
            continue
        if tag_l in ("dibatchjob","dijob","diworkflow","batch_job","job","workflow"):
            current_job = (a.get("name") or a.get("displayname") or current_job).strip()
            continue
        if tag_l == "dischema":
            current_schema = (a.get("name") or a.get("displayname") or current_schema).strip()

        # Sources / Targets
        if tag_l in ("didatabasetablesource","didatabasetabletarget"):
            ds  = (a.get("datastorename") or "").strip()
            sch = (a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl = (a.get("tablename") or "").strip()
            if not (ds and tbl): 
                continue
            proj0, job0, dflow0 = find_context(elem, parent_map)
            dflow0 = dflow0 or current_dflow
            job0   = job0   or df_job_map.get(dflow0, "")
            proj0  = proj0  or df_proj_map.get(dflow0, "")
            role = "source" if "source" in tag_l else "target"
            key = (proj0, job0, dflow0, role, ds, sch, tbl, "", "", 0)
            if key not in seen:
                seen.add(key)
                rows.append(Record(proj0, job0, dflow0, role, ds, sch, tbl, "", "", 0))
            continue

        # Column-level lookup within ui_mapping_text
        if tag_l == "diattribute" and lower(a.get("name","")) == "ui_mapping_text":
            txt = a.get("value") or (elem.text or "")
            # direct lookup(...)
            if "lookup(" in lower(txt):
                ds2, sch2, tbl2 = extract_lookup_from_call(txt, is_ext=False)
                if ds2 and tbl2:
                    proj0, job0, dflow0 = find_context(elem, parent_map)
                    dflow0 = dflow0 or current_dflow
                    job0   = job0   or df_job_map.get(dflow0, "")
                    proj0  = proj0  or df_proj_map.get(dflow0, "")
                    col = find_output_column(elem, parent_map) or ""
                    schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                    if schema_out and col:
                        lookup_pos_entries[(proj0, job0, dflow0, ds2, sch2, tbl2)].append(f"{schema_out}>>{col}")
            # UDF call containing lookups → treat as lookup_ext at this position
            if txt:
                m_udf = UDF_NAME_RE.search(txt)
                if m_udf:
                    udf_name = m_udf.group(1).strip().upper()
                    if udf_name in udf_tables:
                        proj0, job0, dflow0 = find_context(elem, parent_map)
                        dflow0 = dflow0 or current_dflow
                        job0   = job0   or df_job_map.get(dflow0, "")
                        proj0  = proj0  or df_proj_map.get(dflow0, "")
                        schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                        col = find_output_column(elem, parent_map) or ""
                        pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out
                        for (dsu, schu, tblu) in udf_tables[udf_name]:
                            base_key = (proj0, job0, dflow0, dsu, schu, tblu)
                            if pos: lookup_ext_positions[base_key].add(pos)

        # DIExpression hint to link to DIAssignmentStep for lookup_ext
        if tag_l in ("diexpression","diattribute"):
            expr_txt = a.get("expr") or a.get("value") or (elem.text or "")
            if "lookup_ext(" in lower(expr_txt):
                step = nearest_ancestor(elem, parent_map, "diassignmentstep")
                if step is not None:
                    proj0, job0, dflow0 = find_context(elem, parent_map)
                    dflow0 = dflow0 or current_dflow
                    job0   = job0   or df_job_map.get(dflow0, "")
                    proj0  = proj0  or df_proj_map.get(dflow0, "")
                    schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                    pending_ext_by_step[id(step)] = {
                        "schema_out": schema_out, "project": proj0, "job": job0, "dflow": dflow0
                    }

        # FUNCTION_CALL variants (lookup / lookup_ext)
        if tag_l == "function_call":
            fn_name = lower(a.get("name",""))
            ds  = (a.get("tabledatastore") or a.get("datastorename") or "").strip()
            sch = (a.get("tableowner")    or a.get("ownername")     or "").strip()
            tbl = (a.get("tablename")     or "").strip()
            if not (ds and tbl) and ("tabledatastore" in lower(blob) and "tablename" in lower(blob)):
                ds, sch, tbl = extract_lookup_from_attrs_blob(blob)
            if not (ds and tbl):
                ds_c, sch_c, tbl_c = extract_lookup_from_call(blob, is_ext=("lookup_ext" in fn_name))
                ds, sch, tbl = (ds_c or ds), (sch_c or sch), (tbl_c or tbl)

            if ds and tbl:
                proj0, job0, dflow0 = find_context(elem, parent_map)
                dflow0 = dflow0 or current_dflow
                job0   = job0   or df_job_map.get(dflow0, "")
                proj0  = proj0  or df_proj_map.get(dflow0, "")
                schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                col = find_output_column(elem, parent_map) or ""
                pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out

                step = nearest_ancestor(elem, parent_map, "diassignmentstep")
                if step is not None and id(step) in pending_ext_by_step:
                    info = pending_ext_by_step[id(step)]
                    schema_out = info.get("schema_out") or schema_out
                    proj0 = info.get("project") or proj0
                    job0  = info.get("job") or job0
                    dflow0 = info.get("dflow") or dflow0
                    col = find_output_column(elem, parent_map) or col
                    pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out

                base_key = (proj0, job0, dflow0, ds, sch, tbl)

                if "lookup_ext" in fn_name:
                    if pos: lookup_ext_positions[base_key].add(pos)
                elif "lookup" in fn_name:
                    if schema_out and col:
                        lookup_pos_entries[base_key].append(f"{schema_out}>>{col}")

        # ----- Generic expression blobs containing lookup()/lookup_ext() -----
        if "lookup(" in lower(blob) or "lookup_ext(" in lower(blob):
            # column-level lookup(...)
            ds3, sch3, tbl3 = extract_lookup_from_call(blob, is_ext=False)
            if ds3 and tbl3:
                proj0, job0, dflow0 = find_context(elem, parent_map)
                dflow0 = dflow0 or current_dflow
                job0   = job0   or df_job_map.get(dflow0, "")
                proj0  = proj0  or df_proj_map.get(dflow0, "")
                schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                col = find_output_column(elem, parent_map) or ""
                if schema_out and col:
                    lookup_pos_entries[(proj0, job0, dflow0, ds3, sch3, tbl3)].append(f"{schema_out}>>{col}")
            # lookup_ext(...)
            ds4, sch4, tbl4 = extract_lookup_from_call(blob, is_ext=True)
            if ds4 and tbl4:
                proj0, job0, dflow0 = find_context(elem, parent_map)
                dflow0 = dflow0 or current_dflow
                job0   = job0   or df_job_map.get(dflow0, "")
                proj0  = proj0  or df_proj_map.get(dflow0, "")
                schema_out = schema_out_from_DISchema(elem, parent_map, current_schema)
                col = find_output_column(elem, parent_map) or ""
                pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out
                base_key = (proj0, job0, dflow0, ds4, sch4, tbl4)
                if pos: lookup_ext_positions[base_key].add(pos)

    # ---- Emit rows (unique positions, unique counts) ----
    emitted = set()

    # Column-level lookup rows
    for (proj, job, df, ds, sch, tbl), entries in lookup_pos_entries.items():
        unique_positions = dedupe_preserve_order(entries)
        if not unique_positions:
            continue
        lp = ", ".join(unique_positions)
        lookup_used_count = len(unique_positions)
        key = (proj, job, df, "lookup", ds, sch, tbl)
        if key not in emitted:
            emitted.add(key)
            rows.append(Record(proj, job, df, "lookup", ds, sch, tbl, lp, "lookup", lookup_used_count))

    # lookup_ext rows
    for (proj, job, df, ds, sch, tbl), posset in lookup_ext_positions.items():
        uniq = dedupe_preserve_order(list(posset))
        lp = ", ".join(uniq)
        lookup_used_count = len(uniq)
        key = (proj, job, df, "lookup_ext", ds, sch, tbl)
        if key not in emitted:
            emitted.add(key)
            rows.append(Record(proj, job, df, "lookup_ext", ds, sch, tbl, lp, "lookup_ext", lookup_used_count))

    # Sort for readability
    rows.sort(key=lambda r: (
        r.project_name or "", r.job_name or "", r.dataflow_name or "",
        r.role, r.datastore or "", r.schema or "", r.table or "", r.lookup_position or ""
    ))
    return rows

# --------------- main ---------------
def main():
    # <<< CHANGE THIS ONLY >>>
    xml_path = r"C:\path\to\your\export.xml"
    # <<< -------------------- >>>

    if not os.path.isfile(xml_path):
        print(f"File not found: {xml_path}")
        sys.exit(1)

    rows = parse_single_xml(xml_path)
    df = pd.DataFrame([r._asdict() for r in rows]) if rows else pd.DataFrame(columns=Record._fields)

    out_base = os.path.splitext(xml_path)[0] + "_lineage_v2"
    df.to_csv(out_base + ".csv", index=False)
    with pd.ExcelWriter(out_base + ".xlsx") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print("Done.")
    print(f"CSV : {out_base}.csv")
    print(f"XLSX: {out_base}.xlsx")

if __name__ == "__main__":
    main()
