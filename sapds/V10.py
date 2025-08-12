#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, xml.etree.ElementTree as ET
from collections import namedtuple, defaultdict
import pandas as pd

Record = namedtuple("Record", [
    "job_name",
    "dataflow_name",
    "role",               # source / target / lookup / lookup_ext
    "datastore",
    "schema",
    "table",
    "lookup_position",    # For lookup: "SchemaOut>>Col, SchemaOut>>Col2"; empty for lookup_ext/src/tgt
    "used_in_transform"   # "lookup" or "lookup_ext"
])

# ----------------- helpers -----------------
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
        t = lower(strip_ns(anc.tag))
        a = attrs_ci(anc)
        nm = (a.get("name") or a.get("displayname") or "").strip()
        if not dflow and t in ("didataflow","dataflow","dflow"):
            dflow = nm or dflow
        if not job and t in ("dibatchjob","dijob","diworkflow","batch_job","job","workflow"):
            job = nm or job
    return job, dflow

TRANSFORM_TAG_HINTS = ("query","map","transform","lookup","join","key","table_comparison","sort","pivot")
def find_schema_out(elem, parent_map):
    """Nearest ancestor transform's name (Schema Out)."""
    for anc in ancestors(elem, parent_map):
        t = lower(strip_ns(anc.tag))
        if any(h in t for h in TRANSFORM_TAG_HINTS):
            a = attrs_ci(anc)
            return (a.get("schemaoutname") or a.get("schema_out_name") or
                    a.get("name") or a.get("displayname") or "").strip()
    return ""

def find_output_column(elem, parent_map):
    """Closest output column's name."""
    for anc in ancestors(elem, parent_map):
        t = lower(strip_ns(anc.tag))
        if any(x in t for x in ("dicolumn","output_column","column")):
            a = attrs_ci(anc)
            nm = (a.get("name") or a.get("columnname") or "").strip()
            if nm: return nm
    return ""

# Fallback to parse encoded lookup metadata
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
    """Lowercased attr map + concatenated child ATTRIBUTE/CONSTANT values (for regex fallback)."""
    a = attrs_ci(elem)
    parts = [" ".join([f'{k}="{v}"' for k, v in a.items()])]
    for ch in list(elem):
        t = lower(strip_ns(getattr(ch, "tag", "")))
        if t in ("diattribute","attribute","attr","property","prop","param","parameter","constant"):
            val = ch.attrib.get("value")
            if val is None: val = (ch.text or "")
            if val: parts.append(str(val))
    return a, " ".join(parts)

# ----------------- core -----------------
def parse_single_xml(xml_path: str):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] XML parse failed: {e}")
        return []

    parent_map = build_parent_map(root)
    rows, seen = [], set()

    # Collect positions for column-level lookup only (we'll merge later)
    # key = (job, dflow, ds, sch, tbl)
    lookup_pos_entries = defaultdict(list)  # store strings like "Join>>COL" (keep repeats)
    # Track lookup_ext presence (positions left blank)
    lookup_ext_present = set()  # set of keys (job, dflow, ds, sch, tbl)

    current_job = ""
    current_dflow = ""

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue

        tag_l = lower(strip_ns(elem.tag))
        a, blob = gather_attr_text(elem)

        # stream context
        if tag_l in ("didataflow","dataflow","dflow"):
            current_dflow = (a.get("name") or a.get("displayname") or current_dflow).strip()
            continue
        if tag_l in ("dibatchjob","dijob","diworkflow","batch_job","job","workflow"):
            current_job = (a.get("name") or a.get("displayname") or current_job).strip()
            continue

        # sources / targets
        if tag_l in ("didatabasetablesource","didatabasetabletarget"):
            ds  = (a.get("datastorename") or "").strip()
            sch = (a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl = (a.get("tablename") or "").strip()
            if not (ds and tbl): 
                continue
            role = "source" if "source" in tag_l else "target"
            job, dflow = find_context(elem, parent_map)
            job = job or current_job
            dflow = dflow or current_dflow
            key = (job, dflow, role, ds, sch, tbl, "", "")
            if key not in seen:
                seen.add(key)
                rows.append(Record(job, dflow, role, ds, sch, tbl, "", ""))  # lookup_position empty
            continue

        # function calls (lookup / lookup_ext)
        if tag_l == "function_call":
            fn_name = lower(a.get("name",""))
            ds  = (a.get("tabledatastore") or a.get("datastorename") or "").strip()
            sch = (a.get("tableowner")    or a.get("ownername")     or "").strip()
            tbl = (a.get("tablename")     or "").strip()
            if not (ds and tbl) and ("tabledatastore" in lower(blob) and "tablename" in lower(blob)):
                ds, sch, tbl = extract_lookup_from_blob(blob)
            if not (ds and tbl): 
                continue

            job, dflow = find_context(elem, parent_map)
            job = job or current_job
            dflow = dflow or current_dflow
            base_key = (job, dflow, ds, sch, tbl)

            if "lookup_ext" in fn_name:
                # mark presence; position will be blank
                lookup_ext_present.add(base_key)
            elif "lookup" in fn_name:
                # column-level lookup: build "Schema>>Column" and keep ONLY if both parts exist
                schema_out = find_schema_out(elem, parent_map) or ""
                col = find_output_column(elem, parent_map) or ""
                if schema_out and col:
                    lookup_pos_entries[base_key].append(f"{schema_out}>>{col}")
            else:
                # allow lookup context via ancestors if needed
                is_ctx = False
                for anc in ancestors(elem, parent_map):
                    if "lookup" in lower(strip_ns(anc.tag)) or "lookup" in " ".join([lower(v) for v in attrs_ci(anc).values()]):
                        is_ctx = True; break
                if is_ctx:
                    schema_out = find_schema_out(elem, parent_map) or ""
                    col = find_output_column(elem, parent_map) or ""
                    if schema_out and col:
                        lookup_pos_entries[base_key].append(f"{schema_out}>>{col}")

    # Emit lookup rows (one per table+role)
    emitted = set()

    # column-level lookup rows
    for (job, dflow, ds, sch, tbl), entries in lookup_pos_entries.items():
        lp = ", ".join(entries)  # keep repeats as requested
        key = (job, dflow, "lookup", ds, sch, tbl)
        if key not in emitted:
            emitted.add(key)
            rows.append(Record(job, dflow, "lookup", ds, sch, tbl, lp, "lookup"))

    # lookup_ext rows (positions blank)
    for (job, dflow, ds, sch, tbl) in lookup_ext_present:
        key = (job, dflow, "lookup_ext", ds, sch, tbl)
        if key not in emitted:
            emitted.add(key)
            rows.append(Record(job, dflow, "lookup_ext", ds, sch, tbl, "", "lookup_ext"))

    # sort
    rows.sort(key=lambda r: (r.job_name, r.dataflow_name, r.role, r.datastore, r.schema, r.table, r.lookup_position))
    return rows

# ----------------- main -----------------
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
