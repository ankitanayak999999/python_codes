#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, xml.etree.ElementTree as ET
from collections import namedtuple
import pandas as pd

Record = namedtuple("Record", [
    "job_name",
    "dataflow_name",
    "role",                # source / target / lookup
    "datastore",
    "schema",
    "table",
    "used_in_transform"    # NEW: transform/function that uses the lookup
])

# ---------- helpers ----------
def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag).strip() if isinstance(tag, str) else ""
def lower(s): return (s or "").strip().lower()
def build_parent_map(root): return {c: p for p in root.iter() for c in p}
def attrs_ci(elem): return {k.lower(): (v or "") for k, v in getattr(elem, "attrib", {}).items()}

def ancestors(elem, parent_map, max_up=60):
    cur = elem
    for _ in range(max_up):
        if cur is None: break
        yield cur
        cur = parent_map.get(cur)

def find_context_via_ancestors(elem, parent_map):
    job = dflow = ""
    for anc in ancestors(elem, parent_map):
        tag_l = lower(strip_ns(anc.tag))
        a = attrs_ci(anc)
        nm = (a.get("name") or a.get("displayname") or "").strip()
        if not dflow and tag_l in ("didataflow","dataflow","dflow"):
            dflow = nm or dflow
        if not job and tag_l in ("dibatchjob","dijob","diworkflow","batch_job","job","workflow"):
            job = nm or job
    return job, dflow

# Heuristics to capture the transform name housing a lookup
TRANSFORM_TAG_HINTS = (
    "query", "map", "transform", "lookup", "join", "key", "table_comparison", "sort"
)
def find_transform_name(elem, parent_map, elem_attrs):
    """
    For lookups, try:
      1) FUNCTION_CALL@name (e.g., lookup_ext_1)
      2) nearest ancestor whose tag or attributes look like a transform and has @name
    """
    # 1) direct
    if "name" in elem_attrs and elem_attrs["name"]:
        return elem_attrs["name"].strip()

    # 2) ancestors
    for anc in ancestors(elem, parent_map):
        tag = strip_ns(anc.tag)
        a = attrs_ci(anc)
        nm = (a.get("name") or a.get("displayname") or "").strip()
        if not nm:
            continue
        t_low = lower(tag)
        if any(h in t_low for h in TRANSFORM_TAG_HINTS):
            return nm
    return ""

# ---- lookup blob regex fallback ----
LOOKUP_RE = re.compile(
    r'tabledatastore\s*=\s*(?P<q1>[\'"]|&quot;)?(?P<ds>[^\'"&\s]+).*?'
    r'tableowner\s*=\s*(?P<q2>[\'"]|&quot;)?(?P<own>[^\'"&\s]+).*?'
    r'tablename\s*=\s*(?P<q3>[\'"]|&quot;)?(?P<tbl>[^\'"&\s]+)',
    flags=re.IGNORECASE | re.DOTALL
)
def extract_lookup_from_blob(blob: str):
    m = LOOKUP_RE.search(blob or "")
    return (m.group("ds"), m.group("own"), m.group("tbl")) if m else ("","","")

def gather_attr_text(elem):
    """return (attr_map_lower, concatenated text of attrs + child ATTRIBUTE/CONSTANT values)"""
    a = attrs_ci(elem)
    texts = [" ".join([f'{k}="{v}"' for k, v in a.items()])]
    for ch in list(elem):
        t = lower(strip_ns(getattr(ch, "tag", "")))
        if t in ("diattribute","attribute","attr","property","prop","param","parameter","constant"):
            val = ch.attrib.get("value")
            if val is None: val = (ch.text or "")
            if val: texts.append(str(val))
    return a, " ".join(texts)

# ---------- core ----------
def parse_single_xml(xml_path: str):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] XML parse failed: {e}")
        return []

    parent_map = build_parent_map(root)
    rows, seen = [], set()

    # streaming context fallback (helps when some nodes lack clear ancestors)
    current_job = ""
    current_dflow = ""

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue

        tag_l = lower(strip_ns(elem.tag))
        a, text_blob = gather_attr_text(elem)

        # Update context when we encounter job/dataflow nodes
        if tag_l in ("didataflow","dataflow","dflow"):
            current_dflow = (a.get("name") or a.get("displayname") or current_dflow).strip()
        elif tag_l in ("dibatchjob","dijob","diworkflow","batch_job","job","workflow"):
            current_job = (a.get("name") or a.get("displayname") or current_job).strip()

        role = None
        ds = sch = tbl = ""
        used_in_transform = ""

        # --- Normal DB table sources/targets
        if tag_l in ("didatabasetablesource","didatabasetabletarget"):
            ds  = (a.get("datastorename") or "").strip()
            sch = (a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl = (a.get("tablename") or "").strip()
            if ds and tbl:
                role = "source" if "source" in tag_l else "target"
                used_in_transform = ""  # not relevant for plain src/tgt

        # --- Lookup: FUNCTION_CALL with tableDatastore/Owner/Name
        elif tag_l == "function_call":
            ds  = (a.get("tabledatastore") or a.get("datastorename") or "").strip()
            sch = (a.get("tableowner")    or a.get("ownername")     or "").strip()
            tbl = (a.get("tablename")     or "").strip()
            is_lookup = "lookup" in lower(a.get("name","")) or "lookup" in lower(a.get("type",""))
            if ds and tbl and is_lookup:
                role = "lookup"
                used_in_transform = find_transform_name(elem, parent_map, a)

            # Fallback: lookup encoded in blob text
            if role is None and ("tabledatastore" in lower(text_blob) and "tablename" in lower(text_blob)):
                ds2, sch2, tbl2 = extract_lookup_from_blob(text_blob)
                if ds2 and tbl2:
                    role = "lookup"
                    ds, sch, tbl = ds2, sch2, tbl2
                    used_in_transform = find_transform_name(elem, parent_map, a)

        # --- Super‑fallback: any node with encoded lookup + ancestor indicates lookup context
        if role is None and ("tabledatastore" in lower(text_blob) and "tablename" in lower(text_blob)):
            # confirm lookup context from ancestors
            lookup_ctx = False
            for anc in ancestors(elem, parent_map):
                t = lower(strip_ns(anc.tag))
                if "lookup" in t:
                    lookup_ctx = True
                    a_anc = attrs_ci(anc)
                    if not used_in_transform:
                        used_in_transform = (a_anc.get("name") or a_anc.get("displayname") or "").strip()
                    break
                av = " ".join([lower(v) for v in attrs_ci(anc).values() if isinstance(v,str)])
                if "lookup" in av:
                    lookup_ctx = True
                    a_anc = attrs_ci(anc)
                    if not used_in_transform:
                        used_in_transform = (a_anc.get("name") or a_anc.get("displayname") or "").strip()
                    break
            if lookup_ctx:
                ds2, sch2, tbl2 = extract_lookup_from_blob(text_blob)
                if ds2 and tbl2:
                    role = "lookup"
                    ds, sch, tbl = ds2, sch2, tbl2

        if role is None:
            continue

        # Context: prefer ancestors, then streaming fallback
        job, dflow = find_context_via_ancestors(elem, parent_map)
        job = job or current_job
        dflow = dflow or current_dflow

        key = (job, dflow, role, ds, sch, tbl, used_in_transform)
        if key in seen:
            continue
        seen.add(key)
        rows.append(Record(job, dflow, role, ds, sch, tbl, used_in_transform))

    rows.sort(key=lambda r: (r.job_name, r.dataflow_name, r.role, r.datastore, r.schema, r.table, r.used_in_transform))
    return rows

# ---------- main ----------
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
    df.to_csv(out_base + ".csv", index=False)
    with pd.ExcelWriter(out_base + ".xlsx") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print("Done.")
    print(f"CSV : {out_base}.csv")
    print(f"XLSX: {out_base}.xlsx")

if __name__ == "__main__":
    main()
