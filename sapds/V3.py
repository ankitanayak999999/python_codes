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
    return re.sub(r"^\{.*\}", "", tag).strip()

def lower(s): 
    return (s or "").strip().lower()

def build_parent_map(root: ET.Element):
    return {c: p for p in root.iter() for c in p}

def elem_attrs(elem: ET.Element) -> dict:
    return {k.lower(): (v or "") for k, v in getattr(elem, "attrib", {}).items()}

def gather_attributes(elem: ET.Element) -> dict:
    """Merge element attrs + nested DIAttribute/ATTRIBUTE name/value pairs."""
    amap = elem_attrs(elem)
    for ch in list(elem):
        t = lower(strip_ns(getattr(ch, "tag", "")))
        if t in ("diattribute", "attribute", "attr", "property", "prop", "param", "parameter"):
            name = ch.attrib.get("name") or ch.attrib.get("NAME") or ""
            value = ch.attrib.get("value")
            if value is None:
                value = (ch.text or "")
            name = lower(name)
            if name:
                amap[name] = value or ""
    return amap

def ancestor_chain(elem, parent_map, max_up=60):
    chain = []
    cur = elem
    for _ in range(max_up):
        if cur is None:
            break
        chain.append(cur)
        cur = parent_map.get(cur)
    return chain

# -------------------------
# Context (Job / Dataflow)
# -------------------------

def find_context(elem, parent_map):
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
# Role detection
# -------------------------

ROLE_HINTS_SOURCE = ("source", "input", "lookup")
ROLE_HINTS_TARGET = ("target", "output", "sink")

def role_from_text(txt: str) -> str:
    t = lower(txt)
    if any(h in t for h in ROLE_HINTS_TARGET):
        return "target"
    if "lookup" in t:
        return "source"
    if any(h in t for h in ROLE_HINTS_SOURCE):
        return "source"
    return ""

def guess_role(elem, parent_map, amap):
    r = role_from_text(strip_ns(elem.tag))
    if r:
        return r
    keys = " ".join(amap.keys())
    vals = " ".join([lower(v) for v in amap.values() if isinstance(v, str)])
    r = role_from_text(keys) or role_from_text(vals)
    if r:
        return r
    for anc in ancestor_chain(elem, parent_map):
        r = role_from_text(strip_ns(anc.tag))
        if r:
            return r
        a = elem_attrs(anc)
        if a:
            r = role_from_text(" ".join(a.keys())) or role_from_text(" ".join([lower(v) for v in a.values()]))
            if r:
                return r
    return ""

# -------------------------
# Identify & extract
# -------------------------

TABLE_NODE_HINTS = (
    "ditable", "didatabasetablesource", "didatabasetabletarget",
    "table", "sourcetable", "targettable", "lookup_table", "table_ref"
)

# include camelCase variants seen in your screenshot
DS_KEYS     = ("datastore", "datastore_name", "data_store", "datastorename")
SCHEMA_KEYS = ("owner", "schema", "schema_name", "ownername")
TABLE_KEYS  = ("name", "table", "table_name", "object", "object_name", "tablename")
FILE_KEYS   = ("file", "file_name", "filename", "path", "filepath")

def looks_like_table_node(elem, amap) -> bool:
    t = lower(strip_ns(elem.tag))
    if any(h in t for h in TABLE_NODE_HINTS):
        return True
    keys = amap.keys()
    if any(k in keys for k in DS_KEYS + SCHEMA_KEYS + TABLE_KEYS + FILE_KEYS):
        return True
    return False

def first_key(amap: dict, keys) -> str:
    for k in keys:
        if k in amap and str(amap[k]).strip():
            return str(amap[k]).strip().strip('"').strip()
    return ""

def extract_ds_schema_table(amap: dict, elem) -> tuple[str, str, str]:
    ds  = first_key(amap, DS_KEYS)
    sch = first_key(amap, SCHEMA_KEYS)
    tbl = first_key(amap, TABLE_KEYS)
    if not tbl:
        tbl = first_key(amap, FILE_KEYS)
    return ds, sch, tbl

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

        amap = gather_attributes(elem)
        if not looks_like_table_node(elem, amap):
            continue

        ds, sch, tbl = extract_ds_schema_table(amap, elem)
        if not any([ds, sch, tbl]):
            continue

        role = guess_role(elem, parent_map, amap)
        if role not in ("source", "target"):
            continue

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
    if df.empty:
        print("Note: extracted 0 rows. If that happens, grab one more screenshot a level above the DIDatabaseTableTarget/Source to refine role detection context.")

if __name__ == "__main__":
    main()
