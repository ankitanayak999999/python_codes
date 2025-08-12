import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple

import pandas as pd

Record = namedtuple("Record", [
    "job_name",
    "dataflow_name",
    "role",      # source / target
    "datastore",
    "schema",
    "table"
])

# -------------------------
# Helpers
# -------------------------

def strip_ns(tag: str) -> str:
    if not isinstance(tag, str):
        return ""
    return re.sub(r"^\{.*\}", "", tag).strip().lower()

def get_attr_ci_from_map(amap: dict, *keys, default=""):
    for k in keys:
        v = amap.get(k.lower())
        if v is not None:
            return (v or "").strip().strip('"').strip()
    return default

def build_parent_map(root: ET.Element):
    return {c: p for p in root.iter() for c in p}

def gather_attributes(elem: ET.Element) -> dict:
    """
    Build a case-insensitive attribute map from:
      1) element attributes
      2) child nodes like <ATTRIBUTE name="table_name" value="ORDERS"/>
    """
    amap = {k.lower(): (v or "") for k, v in getattr(elem, "attrib", {}).items()}
    for child in list(elem):
        ctag = strip_ns(getattr(child, "tag", ""))
        if ctag in ("attribute", "attr", "property", "prop", "param", "parameter"):
            name = (child.attrib.get("name") or child.attrib.get("NAME") or "").strip()
            # Value may be under 'value' attr or text content
            value = child.attrib.get("value")
            if value is None:
                value = (child.text or "")
            if name:
                amap[name.lower()] = (value or "")
    return amap

def classify_role(tag: str, amap: dict) -> str:
    """
    Decide source/target. Treat LOOKUP as SOURCE. Use tag + attr names/values.
    """
    t = strip_ns(tag)
    if any(x in t for x in ("target", "output")):
        return "target"
    if any(x in t for x in ("source", "input", "lookup")):
        return "source"

    # attribute name hints
    name_keys = "".join(amap.keys())
    if "lookup" in name_keys:
        return "source"
    if "target" in name_keys:
        return "target"
    if "source" in name_keys or "input" in name_keys:
        return "source"
    if "role" in amap:
        r = (amap.get("role") or "").lower()
        if r in ("source", "input", "lookup"):
            return "source"
        if r in ("target", "output"):
            return "target"

    # attribute value hints
    vals = " ".join([str(v).lower() for v in amap.values() if isinstance(v, str)])
    if "lookup" in vals:
        return "source"
    if "target" in vals:
        return "target"
    if "source" in vals or "input" in vals:
        return "source"

    return ""  # unknown

def looks_like_table_node(tag: str, amap: dict) -> bool:
    """
    Identify nodes that reference a physical table/file.
    Now robust to XMLs that put everything into child <ATTRIBUTE> nodes.
    """
    t = strip_ns(tag)
    if any(x in t for x in ("source_table", "target_table", "table", "lookup_table")):
        return True

    keys = set(amap.keys())
    tableish = {"table", "table_name", "name"}
    schemaish = {"schema", "owner", "schema_name"}
    dssh = {"datastore", "datastore_name", "data_store"}
    fileish = {"file", "file_name", "filename", "path"}

    if keys & (tableish | schemaish | dssh | fileish):
        return True

    # some exports name the node like <SOMETHING_LOOKUP ...>
    if "lookup" in t or any("lookup" in k for k in keys):
        return True

    return False

def extract_table_keys(amap: dict, tag: str):
    """
    Pull datastore/schema/table from the attribute map, with sensible fallbacks.
    For flat files, we put the filename/path into 'table'.
    """
    ds = get_attr_ci_from_map(amap, "datastore_name", "datastore", "data_store")
    schema = get_attr_ci_from_map(amap, "schema", "owner", "schema_name")
    table = get_attr_ci_from_map(amap, "table", "table_name")

    # If tag implies a table and only 'name' exists, use that
    t = strip_ns(tag)
    if not table and any(x in t for x in ("source_table", "target_table", "table")):
        table = get_attr_ci_from_map(amap, "name")

    # File-ish fallbacks
    if not table:
        table = get_attr_ci_from_map(amap, "file", "file_name", "filename", "path")

    return (ds or "").strip('" '), (schema or "").strip('" '), (table or "").strip('" ')

def find_context(elem: ET.Element, parent_map):
    """
    Walk ancestors to find job/dataflow names (BATCH_JOB/JOB/WORKFLOW, DATAFLOW/DFLOW).
    """
    job = ""
    dflow = ""
    cur = elem
    for _ in range(50):
        if cur is None:
            break
        tag = strip_ns(cur.tag)
        amap = gather_attributes(cur)
        name = get_attr_ci_from_map(amap, "name", "NAME")
        if not dflow and tag in ("dataflow", "dflow"):
            dflow = (name or dflow).strip()
        if not job and tag in ("batch_job", "job", "workflow"):
            job = (name or job).strip()
        cur = parent_map.get(cur)
    return job, dflow

# -------------------------
# Core extraction
# -------------------------

def parse_single_xml(xml_path: str):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] XML parse failed for {xml_path}: {e}")
        return []

    parent_map = build_parent_map(root)
    rows = []
    seen = set()

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue

        amap = gather_attributes(elem)
        if not looks_like_table_node(elem.tag, amap):
            continue

        role = classify_role(elem.tag, amap)
        if role not in ("source", "target"):
            continue  # strictly keep only source/target

        ds, schema, table = extract_table_keys(amap, elem.tag)
        if not any([ds, schema, table]):
            continue  # nothing meaningful to output

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
    # CHANGE THIS TO YOUR FILE:
    xml_path = r"C:\path\to\your\export.xml"

    if not os.path.isfile(xml_path):
        print(f"File not found: {xml_path}")
        sys.exit(1)

    out_base = os.path.splitext(xml_path)[0] + "_lineage"

    rows = parse_single_xml(xml_path)
    df = pd.DataFrame([r._asdict() for r in rows]) if rows else pd.DataFrame(columns=Record._fields)

    if df.empty:
        # Helpful hint so you know it ran but found no matches
        print("No source/target nodes were detected. If possible later, share a redacted snippet of one dataflow's XML tags around the source/target so I can add that pattern.")
    # Write outputs
    csv_path = out_base + ".csv"
    xlsx_path = out_base + ".xlsx"
    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path) as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print("Done.")
    print(f"CSV : {csv_path}")
    print(f"XLSX: {xlsx_path}")

if __name__ == "__main__":
    main()
