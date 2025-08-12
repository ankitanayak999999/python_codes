import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import namedtuple

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
    """Remove XML namespace."""
    if not isinstance(tag, str):
        return ""
    return re.sub(r"^\{.*\}", "", tag).strip()

def lower(s):
    return (s or "").strip().lower()

def build_parent_map(root: ET.Element):
    return {c: p for p in root.iter() for c in p}

def elem_attrs(elem: ET.Element) -> dict:
    """Lowercased attribute map."""
    return {k.lower(): (v or "") for k, v in getattr(elem, "attrib", {}).items()}

def ancestor_chain(elem, parent_map, max_up=60):
    """Walk up ancestor chain."""
    chain = []
    cur = elem
    for _ in range(max_up):
        if cur is None:
            break
        chain.append(cur)
        cur = parent_map.get(cur)
    return chain

def find_context(elem, parent_map):
    """Find nearest job and dataflow names."""
    job = ""
    dflow = ""
    for anc in ancestor_chain(elem, parent_map):
        tag_l = lower(strip_ns(anc.tag))
        nm = (elem_attrs(anc).get("name") or "").strip()
        if not dflow and tag_l in ("didataflow", "dataflow", "dflow"):
            dflow = nm or dflow
        if not job and tag_l in ("dibatchjob", "dijob", "diworkflow", "batch_job", "job", "workflow"):
            job = nm or job
    return job, dflow

# -------------------------
# Core
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

        tag_l = lower(strip_ns(elem.tag))

        # Only keep DB table source/target
        if tag_l not in ("didatabasetablesource", "didatabasetabletarget"):
            continue

        amap = elem_attrs(elem)
        ds  = amap.get("datastorename", "").strip()
        sch = amap.get("ownername", "").strip()
        tbl = amap.get("tablename", "").strip()

        if not ds or not tbl:
            continue  # skip junk like sysdate, to_char

        role = "source" if "source" in tag_l else "target"

        job, dflow = find_context(elem, parent_map)
        key = (job, dflow, role, ds, sch, tbl)
        if key in seen:
            continue
        seen.add(key)

        rows.append(Record(job, dflow, role, ds, sch, tbl))

    rows.sort(key=lambda r: (r.job_name, r.dataflow_name, r.role, r.datastore, r.schema, r.table))
    return rows

# -------------------------
# Main (hardcoded path)
# -------------------------

def main():
    # CHANGE THIS PATH:
    xml_path = r"C:\path\to\your\export.xml"

    if not os.path.isfile(xml_path):
        print(f"File not found: {xml_path}")
        sys.exit(1)

    out_base = os.path.splitext(xml_path)[0] + "_lineage"

    rows = parse_single_xml(xml_path)
    df = pd.DataFrame([r._asdict() for r in rows]) if rows else pd.DataFrame(columns=Record._fields)

    csv_path  = out_base + ".csv"
    xlsx_path = out_base + ".xlsx"

    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path) as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print("Done.")
    print(f"CSV : {csv_path}")
    print(f"XLSX: {xlsx_path}")

if __name__ == "__main__":
    main()
