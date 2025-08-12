#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import namedtuple
import pandas as pd

Record = namedtuple("Record", [
    "job_name",
    "dataflow_name",
    "role",      # source / target / lookup
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
    return re.sub(r"^\{.*\}", "", tag).strip()

def lower(s):
    return (s or "").strip().lower()

def build_parent_map(root: ET.Element):
    return {c: p for p in root.iter() for c in p}

def attrs_ci(elem: ET.Element) -> dict:
    """Lower-cased attribute map (case-insensitive keys)."""
    return {k.lower(): (v or "") for k, v in getattr(elem, "attrib", {}).items()}

def walk_ancestors(elem, parent_map, max_up=60):
    cur = elem
    for _ in range(max_up):
        if cur is None:
            break
        yield cur
        cur = parent_map.get(cur)

def find_context(elem, parent_map):
    """Return (job_name, dataflow_name) from ancestor chain."""
    job = ""
    dflow = ""
    for anc in walk_ancestors(elem, parent_map):
        tag_l = lower(strip_ns(anc.tag))
        nm = (attrs_ci(anc).get("name") or "").strip()
        if not dflow and tag_l in ("didataflow", "dataflow", "dflow"):
            dflow = nm or dflow
        if not job and tag_l in ("dibatchjob", "dijob", "diworkflow", "batch_job", "job", "workflow"):
            job = nm or job
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
    out = []
    seen = set()

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue

        tag_l = lower(strip_ns(elem.tag))
        a = attrs_ci(elem)

        role = None
        ds = sch = tbl = ""

        # ---- Normal DB sources/targets
        if tag_l in ("didatabasetablesource", "didatabasetabletarget"):
            ds  = (a.get("datastorename") or "").strip()
            sch = (a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl = (a.get("tablename") or "").strip()
            if ds and tbl:
                role = "source" if "source" in tag_l else "target"

        # ---- Lookup tables embedded in FUNCTION_CALL
        # Example from your screenshot:
        # <FUNCTION_CALL name="lookup_ext_1" type="D" tableDatastore="DS_ODS" tableOwner="ETLAFS" tableName="STG_ACCOUNT_ADDRESSES" ... />
        elif tag_l == "function_call":
            ds  = (a.get("tabledatastore") or a.get("datastorename") or "").strip()
            sch = (a.get("tableowner") or a.get("ownername") or "").strip()
            tbl = (a.get("tablename") or "").strip()
            # Consider as lookup if it has a table + "lookup" appears in the function name/type or ancestors
            looks_like_lookup = ("lookup" in lower(a.get("name", ""))) or ("lookup" in lower(a.get("type", "")))
            if not looks_like_lookup:
                # check ancestors quickly
                for anc in walk_ancestors(elem, parent_map):
                    if "lookup" in lower(strip_ns(anc.tag)):
                        looks_like_lookup = True
                        break
            if ds and tbl and looks_like_lookup:
                role = "lookup"

        if role is None:
            continue

        job, dflow = find_context(elem, parent_map)
        key = (job, dflow, role, ds, sch, tbl)
        if key in seen:
            continue
        seen.add(key)
        out.append(Record(job, dflow, role, ds, sch, tbl))

    out.sort(key=lambda r: (r.job_name, r.dataflow_name, r.role, r.datastore, r.schema, r.table))
    return out

# -------------------------
# Main (hardcoded path)
# -------------------------

def main():
    # <<< CHANGE THIS ONLY >>>
    xml_path = r"C:\path\to\your\export.xml"
    # <<< ------------------ >>>

    if not os.path.isfile(xml_path):
        print(f"File not found: {xml_path}")
        sys.exit(1)

    rows = parse_single_xml(xml_path)
    df = pd.DataFrame([r._asdict() for r in rows]) if rows else pd.DataFrame(columns=Record._fields)

    out_base = os.path.splitext(xml_path)[0] + "_lineage"
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
