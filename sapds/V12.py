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
    "used_in_transform",    # "lookup" or "lookup_ext" (blank for src/tgt)
    "lookup_used_count"     # integer count per above rules
])

# ----------------- helpers -----------------
def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag).strip() if isinstance(tag, str) else ""
def lower(s): return (s or "").strip().lower()
def build_parent_map(root): return {c: p for p in root.iter() for c in p}
def attrs_ci(elem): return {k.lower(): (v or "") for k, v in getattr(elem, "attrib", {}).items()}

def ancestors(elem, parent_map, max_up=120):
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

# Schema Out = nearest DISchema name (as in your screenshots)
def schema_out_from_DISchema(elem, parent_map):
    for anc in ancestors(elem, parent_map):
        if lower(strip_ns(anc.tag)) == "dischema":
            nm = (attrs_ci(anc).get("name") or "").strip()
            if nm: return nm
    return ""

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

# lookup( DS.SCHEMA.TABLE , ...)
LOOKUP_CALL_RE = re.compile(
    r'lookup\s*\(\s*"?\s*([A-Za-z0-9_]+)\s*"?\s*\.\s*"?\s*([A-Za-z0-9_]+)\s*"?\s*\.\s*"?\s*([A-Za-z0-9_]+)\s*"?',
    flags=re.IGNORECASE
)
def extract_lookup_from_call(text: str):
    if not text: return ("","","")
    text = re.sub(r'\s*\.\s*', '.', text)
    m = LOOKUP_CALL_RE.search(text)
    if not m: return ("","","")
    return m.group(1), m.group(2), m.group(3)

# attributes-encoded table metadata (fallback)
LOOKUP_ATTR_RE = re.compile(
    r'tabledatastore\s*=\s*(?P<q1>[\'"]|&quot;)?(?P<ds>[^\'"&\s]+).*?'
    r'tableowner\s*=\s*(?P<q2>[\'"]|&quot;)?(?P<own>[^\'"&\s]+).*?'
    r'tablename\s*=\s*(?P<q3>[\'"]|&quot;)?(?P<tbl>[^\'"&\s]+)',
    flags=re.IGNORECASE | re.DOTALL
)
def extract_lookup_from_attrs_blob(blob: str):
    m = LOOKUP_ATTR_RE.search(blob or "")
    return (m.group("ds"), m.group("own"), m.group("tbl")) if m else ("","","")

def gather_attr_text(elem):
    a = attrs_ci(elem)
    parts = [" ".join([f'{k}="{v}"' for k, v in a.items()])]
    for ch in list(elem):
        tag = lower(strip_ns(getattr(ch, "tag", "")))
        if tag in ("diattribute","attribute","attr","property","prop","param","parameter","constant","diexpression"):
            val = ch.attrib.get("value")
            if val is None: val = ch.attrib.get("expr")
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

    # Collect positions
    # column-level lookup: key -> list of "Schema>>Col" (keep repeats)
    lookup_pos_entries = defaultdict(list)                      # (job,dflow,ds,sch,tbl) -> [Schema>>Col,...]
    # transform-level lookup_ext: key -> set of Schema names (dedup)
    lookup_ext_transforms = defaultdict(set)                    # (job,dflow,ds,sch,tbl) -> {Schema,...}

    current_job = ""
    current_dflow = ""

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue

        tag_l = lower(strip_ns(elem.tag))
        a, blob = gather_attr_text(elem)

        # Stream context
        if tag_l in ("didataflow","dataflow","dflow"):
            current_dflow = (a.get("name") or a.get("displayname") or current_dflow).strip()
            continue
        if tag_l in ("dibatchjob","dijob","diworkflow","batch_job","job","workflow"):
            current_job = (a.get("name") or a.get("displayname") or current_job).strip()
            continue

        # Sources / Targets
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
            key = (job, dflow, role, ds, sch, tbl, "", "", 0)
            if key not in seen:
                seen.add(key)
                rows.append(Record(job, dflow, role, ds, sch, tbl, "", "", 0))
            continue

        # Column-level lookup in DIAttribute ui_mapping_text
        if tag_l == "diattribute" and lower(a.get("name","")) == "ui_mapping_text":
            txt = a.get("value") or (elem.text or "")
            if "lookup(" in lower(txt):
                ds2, sch2, tbl2 = extract_lookup_from_call(txt)
                if ds2 and tbl2:
                    job, dflow = find_context(elem, parent_map)
                    job = job or current_job
                    dflow = dflow or current_dflow
                    col = find_output_column(elem, parent_map) or ""
                    schema_out = schema_out_from_DISchema(elem, parent_map) or ""
                    if schema_out and col:
                        lookup_pos_entries[(job, dflow, ds2, sch2, tbl2)].append(f"{schema_out}>>{col}")

        # FUNCTION_CALL variants (lookup / lookup_ext)
        if tag_l == "function_call":
            fn_name = lower(a.get("name",""))
            ds  = (a.get("tabledatastore") or a.get("datastorename") or "").strip()
            sch = (a.get("tableowner")    or a.get("ownername")     or "").strip()
            tbl = (a.get("tablename")     or "").strip()
            if not (ds and tbl) and ("tabledatastore" in lower(blob) and "tablename" in lower(blob)):
                ds, sch, tbl = extract_lookup_from_attrs_blob(blob)
            if ds and tbl:
                job, dflow = find_context(elem, parent_map)
                job = job or current_job
                dflow = dflow or current_dflow
                base_key = (job, dflow, ds, sch, tbl)
                if "lookup_ext" in fn_name:
                    schema_out = schema_out_from_DISchema(elem, parent_map) or ""
                    if schema_out:
                        lookup_ext_transforms[base_key].add(schema_out)
                    else:
                        lookup_ext_transforms[base_key]
                elif "lookup" in fn_name:
                    schema_out = schema_out_from_DISchema(elem, parent_map) or ""
                    col = find_output_column(elem, parent_map) or ""
                    if schema_out and col:
                        lookup_pos_entries[base_key].append(f"{schema_out}>>{col}")

        # Any expression blob containing lookup(DS.SCH.TBL, ...)
        if "lookup(" in lower(blob):
            ds3, sch3, tbl3 = extract_lookup_from_call(blob)
            if ds3 and tbl3:
                job, dflow = find_context(elem, parent_map)
                job = job or current_job
                dflow = dflow or current_dflow
                schema_out = schema_out_from_DISchema(elem, parent_map) or ""
                col = find_output_column(elem, parent_map) or ""
                if schema_out and col:
                    lookup_pos_entries[(job, dflow, ds3, sch3, tbl3)].append(f"{schema_out}>>{col}")

    # Emit lookup rows (one per table+role)
    emitted = set()

    # Column-level lookup rows
    for (job, dflow, ds, sch, tbl), entries in lookup_pos_entries.items():
        if not entries: 
            continue
        lp = ", ".join(entries)                 # repeats allowed
        lookup_used_count = len(entries)
        key = (job, dflow, "lookup", ds, sch, tbl)
        if key not in emitted:
            emitted.add(key)
            rows.append(Record(job, dflow, "lookup", ds, sch, tbl, lp, "lookup", lookup_used_count))

    # Transform-level lookup_ext rows
    for (job, dflow, ds, sch, tbl), schemas in lookup_ext_transforms.items():
        lp = ", ".join(sorted(s for s in schemas if s))   # e.g., "Join, Calc"
        lookup_used_count = len([s for s in schemas if s])
        key = (job, dflow, "lookup_ext", ds, sch, tbl)
        if key not in emitted:
            emitted.add(key)
            rows.append(Record(job, dflow, "lookup_ext", ds, sch, tbl, lp, "lookup_ext", lookup_used_count))

    # Sort for readability
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
