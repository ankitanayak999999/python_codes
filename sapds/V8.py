#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, xml.etree.ElementTree as ET
from collections import namedtuple
import pandas as pd

Record = namedtuple("Record", [
    "job_name",
    "dataflow_name",
    "role",               # source / target / lookup / lookup_ext
    "datastore",
    "schema",
    "table",
    "schema_out",         # parent transform (e.g., Join)
    "lookup_position",    # ONLY for column-level lookup(...) ; blank otherwise
    "used_in_transform"   # function/transform name (lookup / lookup_ext / etc.)
])

# ---------- helpers ----------
def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag).strip() if isinstance(tag, str) else ""
def lower(s): return (s or "").strip().lower()
def build_parent_map(root): return {c: p for p in root.iter() for c in p}
def attrs_ci(elem): return {k.lower(): (v or "") for k, v in getattr(elem, "attrib", {}).items()}

def ancestors(elem, parent_map, max_up=80):
    cur = elem
    for _ in range(max_up):
        if cur is None: break
        yield cur
        cur = parent_map.get(cur)

def find_context(elem, parent_map):
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

TRANSFORM_TAG_HINTS = ("query","map","transform","lookup","join","key","table_comparison","sort","pivot")

def find_schema_out(elem, parent_map):
    """Always return nearest ancestor transform's name as Schema Out."""
    for anc in ancestors(elem, parent_map):
        t_low = lower(strip_ns(anc.tag))
        if any(h in t_low for h in TRANSFORM_TAG_HINTS):
            a = attrs_ci(anc)
            return (a.get("schemaoutname") or a.get("schema_out_name") or
                    a.get("name") or a.get("displayname") or "").strip()
    return ""

# Column name for column-level lookup(...) only
def find_output_column(elem, parent_map):
    for anc in ancestors(elem, parent_map):
        tag_l = lower(strip_ns(anc.tag))
        if any(x in tag_l for x in ("dicolumn","output_column","column")):
            a = attrs_ci(anc)
            nm = (a.get("name") or a.get("columnname") or "").strip()
            if nm:
                return nm
    return ""

# regex fallback to read encoded lookup table metadata
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
    """attr map + concatenated child ATTRIBUTE/CONSTANT values (for regex fallback)."""
    a = attrs_ci(elem)
    parts = [" ".join([f'{k}="{v}"' for k, v in a.items()])]
    for ch in list(elem):
        t = lower(strip_ns(getattr(ch, "tag", "")))
        if t in ("diattribute","attribute","attr","property","prop","param","parameter","constant"):
            val = ch.attrib.get("value")
            if val is None: val = (ch.text or "")
            if val: parts.append(str(val))
    return a, " ".join(parts)

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

    current_job = ""
    current_dflow = ""

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue

        tag_l = lower(strip_ns(elem.tag))
        a, blob = gather_attr_text(elem)

        # streaming context
        if tag_l in ("didataflow","dataflow","dflow"):
            current_dflow = (a.get("name") or a.get("displayname") or current_dflow).strip()
        elif tag_l in ("dibatchjob","dijob","diworkflow","batch_job","job","workflow"):
            current_job = (a.get("name") or a.get("displayname") or current_job).strip()

        role = None
        ds = sch = tbl = ""
        schema_out = ""
        lookup_position = ""
        used_in = ""

        # ---- normal DB tables
        if tag_l in ("didatabasetablesource","didatabasetabletarget"):
            ds  = (a.get("datastorename") or "").strip()
            sch = (a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl = (a.get("tablename") or "").strip()
            if ds and tbl:
                role = "source" if "source" in tag_l else "target"
                schema_out = ""  # not applicable

        # ---- FUNCTION_CALL (lookup or lookup_ext)
        elif tag_l == "function_call":
            fn_name = lower(a.get("name",""))
            ds  = (a.get("tabledatastore") or a.get("datastorename") or "").strip()
            sch = (a.get("tableowner")    or a.get("ownername")     or "").strip()
            tbl = (a.get("tablename")     or "").strip()

            # column-level lookup(...)
            if "lookup_ext" not in fn_name and "lookup" in fn_name and ds and tbl:
                role = "lookup"
                used_in = a.get("name","lookup").strip()
                schema_out = find_schema_out(elem, parent_map)
                lookup_position = find_output_column(elem, parent_map)  # ONLY here

            # transform-level lookup_ext (one row, NO per-column duplication)
            elif "lookup_ext" in fn_name and ds and tbl:
                role = "lookup_ext"
                used_in = a.get("name","lookup_ext").strip()
                schema_out = find_schema_out(elem, parent_map)
                lookup_position = ""  # per your requirement

            # fallback: encoded metadata in blob + lookup context
            if role is None and ("tabledatastore" in lower(blob) and "tablename" in lower(blob)):
                # confirm lookup context
                is_lookup_ctx = ("lookup" in fn_name)
                if not is_lookup_ctx:
                    for anc in ancestors(elem, parent_map):
                        if "lookup" in lower(strip_ns(anc.tag)) or "lookup" in " ".join([lower(v) for v in attrs_ci(anc).values()]):
                            is_lookup_ctx = True
                            break
                if is_lookup_ctx:
                    ds2, sch2, tbl2 = extract_lookup_from_blob(blob)
                    if ds2 and tbl2:
                        role = "lookup" if "lookup_ext" not in fn_name else "lookup_ext"
                        ds, sch, tbl = ds2, sch2, tbl2
                        used_in = a.get("name", role).strip()
                        schema_out = find_schema_out(elem, parent_map)
                        lookup_position = "" if role == "lookup_ext" else find_output_column(elem, parent_map)

        if role is None:
            continue

        # context with fallback
        job, dflow = find_context(elem, parent_map)
        job = job or current_job
        dflow = dflow or current_dflow

        key = (job, dflow, role, ds, sch, tbl, schema_out, lookup_position, used_in)
        if key in seen:
            continue
        seen.add(key)
        rows.append(Record(job, dflow, role, ds, sch, tbl, schema_out, lookup_position, used_in))

    rows.sort(key=lambda r: (r.job_name, r.dataflow_name, r.role, r.datastore, r.schema, r.table, r.schema_out, r.lookup_position))
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
