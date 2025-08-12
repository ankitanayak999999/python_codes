import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple

import pandas as pd

Record = namedtuple("Record", [
    "job_name",
    "dataflow_name",
    "role",  # source / target
    "datastore",
    "schema",
    "table"
])

# -------------------------
# Helpers
# -------------------------

def strip_ns(tag: str) -> str:
    if tag is None:
        return ""
    return re.sub(r"^\{.*\}", "", tag).strip().lower()

def get_attr_ci(elem: ET.Element, *keys, default=""):
    if elem is None or not hasattr(elem, "attrib"):
        return default
    amap = {k.lower(): v for k, v in elem.attrib.items()}
    for k in keys:
        if k.lower() in amap:
            return (amap[k.lower()] or "").strip().strip('"').strip()
    return default

def any_attr_name_contains(elem: ET.Element, needle: str) -> bool:
    n = needle.lower()
    for k in elem.attrib.keys():
        if n in k.lower():
            return True
    return False

def any_attr_value_contains(elem: ET.Element, needle: str) -> bool:
    n = needle.lower()
    for v in elem.attrib.values():
        if isinstance(v, str) and n in v.lower():
            return True
    return False

def classify_role(elem: ET.Element, tag_hint: str) -> str:
    """
    Decide source/target. Treat LOOKUPs as SOURCE.
    Uses tag names and attribute names/values for hints.
    """
    t = strip_ns(tag_hint or elem.tag)

    # Direct tag hints
    if "target" in t or "output" in t or t.endswith("_target"):
        return "target"
    if ("source" in t or "input" in t or "lookup" in t or t.endswith("_source")):
        return "source"

    # Attribute name hints
    if any_attr_name_contains(elem, "lookup"):
        return "source"
    if any_attr_name_contains(elem, "source"):
        return "source"
    if any_attr_name_contains(elem, "target") or any_attr_name_contains(elem, "output"):
        return "target"
    if any_attr_name_contains(elem, "role"):
        role = get_attr_ci(elem, "role")
        if role.lower() in ("source", "input", "lookup"):
            return "source"
        if role.lower() in ("target", "output"):
            return "target"

    # Attribute value hints
    if any_attr_value_contains(elem, "lookup"):
        return "source"
    if any_attr_value_contains(elem, "source"):
        return "source"
    if any_attr_value_contains(elem, "target") or any_attr_value_contains(elem, "output"):
        return "target"

    return ""  # unknown

def build_parent_map(root: ET.Element):
    return {c: p for p in root.iter() for c in p}

def find_context(elem, parent_map):
    """
    Walk ancestors to find job/dataflow names.
    Recognizes BATCH_JOB/JOB/WORKFLOW and DATAFLOW/DFLOW.
    """
    job = ""
    dflow = ""
    cur = elem
    for _ in range(50):
        if cur is None:
            break
        tag = strip_ns(cur.tag)
        name = get_attr_ci(cur, "name", "NAME", default="").strip()
        if not dflow and tag in ("dataflow", "dflow"):
            dflow = name or dflow
        if not job and tag in ("batch_job", "job", "workflow"):
            job = name or job
        cur = parent_map.get(cur)
    return job, dflow

def extract_table_keys(elem: ET.Element):
    ds = get_attr_ci(elem, "datastore_name", "datastore", "data_store")
    schema = get_attr_ci(elem, "schema", "owner", "schema_name")
    table = get_attr_ci(elem, "table", "table_name")
    tag = strip_ns(elem.tag)
    if not table and tag in ("source_table", "target_table", "table"):
        table = get_attr_ci(elem, "name")
    if not table:
        file_like = get_attr_ci(elem, "file", "file_name", "filename", "path")
        if file_like:
            table = file_like
    return (ds or "").strip().strip('"'), (schema or "").strip().strip('"'), (table or "").strip().strip('"')

def looks_like_table_node(elem: ET.Element) -> bool:
    tag = strip_ns(elem.tag)
    if any(x in tag for x in ("source_table", "target_table", "table")):
        return True
    keys = [k.lower() for k in elem.attrib.keys()]
    if any(k in keys for k in ("table", "table_name", "owner", "schema", "schema_name", "datastore", "datastore_name")):
        return True
    if any(k in keys for k in ("file", "file_name", "filename", "path")):
        return True
    if "lookup" in tag:
        return True
    if any("lookup" in k for k in keys):
        return True
    return False

# -------------------------
# Core extraction
# -------------------------

def parse_single_xml(xml_path: str):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[WARN] XML parse failed for {xml_path}: {e}")
        return []

    parent_map = build_parent_map(root)
    rows = []
    seen = set()

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        if not looks_like_table_node(elem):
            continue
        role = classify_role(elem, elem.tag)
        if role not in ("source", "target"):
            continue
        ds, schema, table = extract_table_keys(elem)
        if not any([ds, schema, table]):
            continue
        job, dflow = find_context(elem, parent_map)
        key = (job, dflow, role, ds, schema, table)
        if key in seen:
            continue
        seen.add(key)
        rows.append(Record(job, dflow, role, ds, schema, table))

    rows.sort(key=lambda r: (r.job_name, r.dataflow_name, r.role, r.datastore, r.schema, r.table))
    return rows

# -------------------------
# Main (hardcoded path)
# -------------------------

def main():
    # CHANGE THIS PATH ONLY:
    xml_path = r"C:\path\to\your\export.xml"

    if not os.path.isfile(xml_path):
        print(f"File not found: {xml_path}")
        sys.exit(1)

    out_base = os.path.splitext(xml_path)[0] + "_lineage"

    rows = parse_single_xml(xml_path)
    df = pd.DataFrame([r._asdict() for r in rows]) if rows else pd.DataFrame(columns=Record._fields)

    # Always write CSV
    csv_path = out_base + ".csv"
    df.to_csv(csv_path, index=False)

    # Always write Excel
    xlsx_path = out_base + ".xlsx"
    with pd.ExcelWriter(xlsx_path) as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print("Done.")
    print(f"CSV : {csv_path}")
    print(f"XLSX: {xlsx_path}")

if __name__ == "__main__":
    main()
