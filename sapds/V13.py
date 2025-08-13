#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, xml.etree.ElementTree as ET
from collections import namedtuple, defaultdict
import pandas as pd

Record = namedtuple("Record", [
    "job_name",
    "dataflow_name",
    "role",                 # source / target / lookup / lookup_ext
    "datastore",
    "schema",
    "table",
    "lookup_position",      # lookup: "Schema>>Col, ..."; lookup_ext: "Schema, ..."; src/tgt: ""
    "used_in_transform",    # "lookup" / "lookup_ext" (blank for src/tgt)
    "lookup_used_count"     # UNIQUE positions count
])

# ---------------- helpers ----------------
def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag).strip() if isinstance(tag, str) else ""
def lower(s): return (s or "").strip().lower()
def build_parent_map(root): return {c: p for p in root.iter() for c in p}
def attrs_ci(elem): return {k.lower(): (v or "") for k, v in getattr(elem, "attrib", {}).items()}

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
    job = dflow = ""
    for anc in ancestors(elem, parent_map):
        t = lower(strip_ns(anc.tag))
        a = attrs_ci(anc)
        nm = (a.get("name") or a.get("displayname") or "").strip()
        if not dflow and t in ("didataflow","dataflow","dflow"):
            dflow = nm or dflow
        if not job and t in ("dibatchjob","dijob","diworkflow","batch_job","job","workflow"):
            job = nm or job
    return job, dflow

# Schema Out = nearest DISchema name (fallback to streaming)
def schema_out_from_DISchema(elem, parent_map, fallback=""):
    for anc in ancestors(elem, parent_map):
        if lower(strip_ns(anc.tag)) == "dischema":
            nm = (attrs_ci(anc).get("name") or "").strip()
            if nm: return nm
    return fallback or ""

# Nearest output column (DIElement name)
def find_output_column(elem, parent_map):
    if lower(strip_ns(elem.tag)) == "dielement":
        nm = (attrs_ci(elem).get("name") or "").strip()
        if nm: return nm
    for anc in ancestors(elem, parent_map):
        if lower(strip_ns(anc.tag)) == "dielement":
            nm = (attrs_ci(anc).get("name") or "").strip()
            if nm: return nm
    return ""

# Pull all text/attrs within a node (for scanning)
def collect_text(node):
    buf = []
    if hasattr(node, "attrib"):
        for _, v in node.attrib.items():
            if v: buf.append(str(v))
    if node.text: buf.append(node.text)
    for ch in list(node):
        buf.append(collect_text(ch))
        if ch.tail: buf.append(ch.tail)
    return " ".join([b for b in buf if b])

# Regexes
DOT_NORMALIZE = re.compile(r"\s*\.\s*")
LOOKUP_CALL_RE     = re.compile(r'lookup\s*\(\s*"?\s*([A-Za-z0-9_]+)\s*"?\.\s*"?\s*([A-Za-z0-9_]+)\s*"?\.\s*"?\s*([A-Za-z0-9_]+)\s*"?', re.IGNORECASE)
LOOKUP_EXT_CALL_RE = re.compile(r'lookup_ext\s*\(\s*"?\s*([A-Za-z0-9_]+)\s*"?\.\s*"?\s*([A-Za-z0-9_]+)\s*"?\.\s*"?\s*([A-Za-z0-9_]+)\s*"?', re.IGNORECASE)

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

# UDF indexing: map UDF name -> set of (ds, sch, tbl) used inside the UDF
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

# --------------- core ---------------
def parse_single_xml(xml_path: str):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] XML parse failed: {e}")
        return []

    parent_map = build_parent_map(root)
    udf_tables = build_udf_lookup_index(root)

    rows, seen = [], set()

    # Collect positions
    # column-level lookup: key -> list of "Schema>>Col"
    lookup_pos_entries = defaultdict(list)          # (job,dflow,ds,sch,tbl) -> [Schema>>Col,...]
    # transform-level lookup_ext: key -> set of Schema names (dedup)
    lookup_ext_transforms = defaultdict(set)        # (job,dflow,ds,sch,tbl) -> {Schema,...}

    # streaming context + linkage for assignment steps
    current_job = ""
    current_dflow = ""
    current_schema_out = ""
    pending_ext_by_step = {}  # id(<DIAssignmentStep>) -> {"schema_out","job","dflow"}

    for elem in root.iter():
        if not isinstance(elem.tag, str): continue

        tag_l = lower(strip_ns(elem.tag))
        a = attrs_ci(elem)
        # gather blob text from children (for generic scanning)
        blob_parts = [" ".join([f'{k}="{v}"' for k, v in a.items()])]
        for ch in list(elem):
            t = lower(strip_ns(getattr(ch, "tag", "")))
            if t in ("diattribute","attribute","attr","property","prop","param","parameter","constant","diexpression"):
                val = ch.attrib.get("value") or ch.attrib.get("expr") or (ch.text or "")
                if val: blob_parts.append(str(val))
        blob = " ".join(blob_parts)

        # stream context
        if tag_l in ("didataflow","dataflow","dflow"):
            current_dflow = (a.get("name") or a.get("displayname") or current_dflow).strip()
            continue
        if tag_l in ("dibatchjob","dijob","diworkflow","batch_job","job","workflow"):
            current_job = (a.get("name") or a.get("displayname") or current_job).strip()
            continue
        if tag_l == "dischema":
            current_schema_out = (a.get("name") or a.get("displayname") or current_schema_out).strip()

        # Sources / Targets
        if tag_l in ("didatabasetablesource","didatabasetabletarget"):
            ds  = (a.get("datastorename") or "").strip()
            sch = (a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl = (a.get("tablename") or "").strip()
            if not (ds and tbl): continue
            role = "source" if "source" in tag_l else "target"
            job, dflow = find_context(elem, parent_map)
            job = job or current_job; dflow = dflow or current_dflow
            key = (job, dflow, role, ds, sch, tbl, "", "", 0)
            if key not in seen:
                seen.add(key)
                rows.append(Record(job, dflow, role, ds, sch, tbl, "", "", 0))
            continue

        # ----- Column-level lookup within DIAttribute ui_mapping_text -----
        if tag_l == "diattribute" and lower(a.get("name","")) == "ui_mapping_text":
            txt = a.get("value") or (elem.text or "")
            # direct lookup(...)
            if "lookup(" in lower(txt):
                ds2, sch2, tbl2 = extract_lookup_from_call(txt, is_ext=False)
                if ds2 and tbl2:
                    job, dflow = find_context(elem, parent_map)
                    job = job or current_job; dflow = dflow or current_dflow
                    col = find_output_column(elem, parent_map) or ""
                    schema_out = schema_out_from_DISchema(elem, parent_map, current_schema_out)
                    if schema_out and col:
                        lookup_pos_entries[(job, dflow, ds2, sch2, tbl2)].append(f"{schema_out}>>{col}")
            # UDF call → attach UDF's internal lookup tables as lookup_ext at this schema
            if txt:
                m_udf = UDF_NAME_RE.search(txt)
                if m_udf:
                    udf_name = m_udf.group(1).strip().upper()
                    if udf_name in udf_tables:
                        job, dflow = find_context(elem, parent_map)
                        job = job or current_job; dflow = dflow or current_dflow
                        schema_out = schema_out_from_DISchema(elem, parent_map, current_schema_out)
                        for (dsu, schu, tblu) in udf_tables[udf_name]:
                            base_key = (job, dflow, dsu, schu, tblu)
                            if schema_out:
                                lookup_ext_transforms[base_key].add(schema_out)
                            else:
                                lookup_ext_transforms[base_key]
            # fall through

        # ----- Link DIExpression lookup_ext(...) to its DIAssignmentStep (to get schema/job/df) -----
        if tag_l in ("diexpression","diattribute"):
            expr_txt = a.get("expr") or a.get("value") or (elem.text or "")
            if "lookup_ext(" in lower(expr_txt):
                step = nearest_ancestor(elem, parent_map, "diassignmentstep")
                if step is not None:
                    job, dflow = find_context(elem, parent_map)
                    job = job or current_job; dflow = dflow or current_dflow
                    schema_out = schema_out_from_DISchema(elem, parent_map, current_schema_out)
                    pending_ext_by_step[id(step)] = {"schema_out": schema_out, "job": job, "dflow": dflow}

        # ----- FUNCTION_CALL variants (lookup / lookup_ext) -----
        if tag_l == "function_call":
            fn_name = lower(a.get("name",""))
            ds  = (a.get("tabledatastore") or a.get("datastorename") or "").strip()
            sch = (a.get("tableowner")    or a.get("ownername")     or "").strip()
            tbl = (a.get("tablename")     or "").strip()
            if not (ds and tbl) and ("tabledatastore" in lower(blob) and "tablename" in lower(blob)):
                ds, sch, tbl = extract_lookup_from_attrs_blob(blob)
            if not (ds and tbl):
                # maybe only visible as lookup(...) text:
                ds_c, sch_c, tbl_c = extract_lookup_from_call(blob, is_ext=("lookup_ext" in fn_name))
                ds, sch, tbl = (ds_c or ds), (sch_c or sch), (tbl_c or tbl)

            if ds and tbl:
                # default context
                job, dflow = find_context(elem, parent_map)
                job = job or current_job; dflow = dflow or current_dflow
                schema_out = schema_out_from_DISchema(elem, parent_map, current_schema_out)

                # if inside a DIAssignmentStep that we saw earlier, override with that info
                step = nearest_ancestor(elem, parent_map, "diassignmentstep")
                if step is not None and id(step) in pending_ext_by_step:
                    info = pending_ext_by_step[id(step)]
                    schema_out = info.get("schema_out") or schema_out
                    job = info.get("job") or job
                    dflow = info.get("dflow") or dflow

                base_key = (job, dflow, ds, sch, tbl)

                if "lookup_ext" in fn_name:
                    if schema_out:
                        lookup_ext_transforms[base_key].add(schema_out)
                    else:
                        lookup_ext_transforms[base_key]
                elif "lookup" in fn_name:
                    col = find_output_column(elem, parent_map) or ""
                    if schema_out and col:
                        lookup_pos_entries[base_key].append(f"{schema_out}>>{col}")

        # ----- Generic expression blobs containing lookup()/lookup_ext() -----
        if "lookup(" in lower(blob) or "lookup_ext(" in lower(blob):
            # column-level lookup(...)
            ds3, sch3, tbl3 = extract_lookup_from_call(blob, is_ext=False)
            if ds3 and tbl3:
                job, dflow = find_context(elem, parent_map)
                job = job or current_job; dflow = dflow or current_dflow
                schema_out = schema_out_from_DISchema(elem, parent_map, current_schema_out)
                col = find_output_column(elem, parent_map) or ""
                if schema_out and col:
                    lookup_pos_entries[(job, dflow, ds3, sch3, tbl3)].append(f"{schema_out}>>{col}")
            # transform-level lookup_ext(...)
            ds4, sch4, tbl4 = extract_lookup_from_call(blob, is_ext=True)
            if ds4 and tbl4:
                job, dflow = find_context(elem, parent_map)
                job = job or current_job; dflow = dflow or current_dflow
                schema_out = schema_out_from_DISchema(elem, parent_map, current_schema_out)
                base_key = (job, dflow, ds4, sch4, tbl4)
                if schema_out:
                    lookup_ext_transforms[base_key].add(schema_out)
                else:
                    lookup_ext_transforms[base_key]

    # ---- Emit rows (unique positions, unique counts) ----
    emitted = set()

    # Column-level lookup rows
    for (job, dflow, ds, sch, tbl), entries in lookup_pos_entries.items():
        unique_positions = dedupe_preserve_order(entries)
        if not unique_positions: 
            continue
        lp = ", ".join(unique_positions)
        lookup_used_count = len(unique_positions)
        key = (job, dflow, "lookup", ds, sch, tbl)
        if key not in emitted:
            emitted.add(key)
            rows.append(Record(job, dflow, "lookup", ds, sch, tbl, lp, "lookup", lookup_used_count))

    # Transform-level lookup_ext rows
    for (job, dflow, ds, sch, tbl), schemas in lookup_ext_transforms.items():
        uniq = dedupe_preserve_order(list(schemas))
        lp = ", ".join(uniq)
        lookup_used_count = len(uniq)
        key = (job, dflow, "lookup_ext", ds, sch, tbl)
        if key not in emitted:
            emitted.add(key)
            rows.append(Record(job, dflow, "lookup_ext", ds, sch, tbl, lp, "lookup_ext", lookup_used_count))

    # Sort for readability
    rows.sort(key=lambda r: (r.job_name, r.dataflow_name, r.role, r.datastore, r.schema, r.table, r.lookup_position))
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

    out_base = os.path.splitext(xml_path)[0] + "_lineage"
    df.to_csv(out_base + ".csv", index=False)
    with pd.ExcelWriter(out_base + ".xlsx") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print("Done.")
    print(f"CSV : {out_base}.csv")
    print(f"XLSX: {out_base}.xlsx")

if __name__ == "__main__":
    main()
