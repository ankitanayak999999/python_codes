#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from collections import defaultdict
from xml.etree import ElementTree as ET
import pandas as pd

# ---------- helpers ----------
def strip_ns(tag: str) -> str:
    if tag and tag.startswith("{"):
        return tag.split("}", 1)[1].lower()
    return (tag or "").lower()

def low(s): return (s or "").lower()
def clean(s: str) -> str:
    s = (s or "").strip().strip('"').strip("'")
    # drop surrounding [] or {}
    if len(s) >= 2 and s[0] in "[{" and s[-1] in "]}":
        s = s[1:-1]
    # remove stray brackets and collapse spaces
    s = s.replace("[", "").replace("]", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_parent_map(root):
    pm = {}
    for p in root.iter():
        for c in p:
            pm[c] = p
    return pm

def nearest_schema_out(elem, parent_map):
    cur = elem
    for _ in range(80):
        if cur is None: break
        if strip_ns(cur.tag) == "dischema":
            nm = clean(cur.attrib.get("name") or "")
            return nm or "Join"
        cur = parent_map.get(cur)
    return "Join"

def nearest_output_column(elem, parent_map):
    cur = elem
    for _ in range(80):
        if cur is None: break
        if strip_ns(cur.tag) == "dielement":
            nm = clean(cur.attrib.get("name") or "")
            if nm: return nm
        cur = parent_map.get(cur)
    return ""

# ---------- DS / schema / table parsing ----------
RGX_TRIPLE = re.compile(
    r"\b(?:lookup|lookup[_ ]?ext)\s*\(\s*([^\s,()\[\]]+)\s*,\s*([^\s,()\[\]]+)\s*,\s*([^\s,()\[\]]+)",
    re.I,
)
RGX_BRACKET = re.compile(r"\[\s*([^\s\].]+)\.([^\s\].]+)\.([^\s\].]+)\s*\]", re.I)

def extract_dst_from_text(txt: str):
    if not txt: return ("", "", "")
    m = RGX_TRIPLE.search(txt)
    if m:
        return clean(m.group(1)), clean(m.group(2)), clean(m.group(3))
    m = RGX_BRACKET.search(txt)
    if m:
        return clean(m.group(1)), clean(m.group(2)), clean(m.group(3))
    return ("", "", "")

def dst_from_function_call(fc):
    ds  = clean(fc.attrib.get("tabledatastore") or "")
    sch = clean(fc.attrib.get("tableowner") or "")
    tbl = clean(fc.attrib.get("tablename") or "")
    if ds and tbl:
        return ds, sch, tbl
    txt = " ".join([fc.attrib.get("expr",""), "".join(fc.itertext())])
    ds2, sch2, tbl2 = extract_dst_from_text(txt)
    return ds or ds2, sch or sch2, tbl or tbl2

# ---------- DIScriptFunction indexing ----------
def index_user_functions(root):
    """Return:
       calls_by_fn: UPPER_NAME -> list[{role, ds, sch, tbl}]
       fn_names: set(UPPER_NAME)
    """
    calls_by_fn = defaultdict(list)
    fn_names = set()
    for fn in root.iter():
        if strip_ns(fn.tag) != "discriptfunction":
            continue
        fname = clean(fn.attrib.get("name",""))
        if not fname: continue
        key = fname.upper()
        fn_names.add(key)

        # structured FUNCTION_CALLs inside the function
        for fc in fn.iter():
            if strip_ns(fc.tag) != "function_call": continue
            nm = low(fc.attrib.get("name"))
            if nm not in ("lookup","lookup_ext"): continue
            ds, sch, tbl = dst_from_function_call(fc)
            if not (ds and tbl):
                ds, sch, tbl = extract_dst_from_text("".join(fc.itertext()))
            if not (ds and tbl): continue
            calls_by_fn[key].append({
                "role": "lookup_ext" if nm == "lookup_ext" else "lookup",
                "ds": ds, "sch": sch, "tbl": tbl
            })

        # fallback: plain text inside the function
        txt = "".join(fn.itertext())
        if re.search(r"\blookup_ext\s*\(", txt, flags=re.I) or re.search(r"\blookup\s*\(", txt, flags=re.I):
            ds, sch, tbl = extract_dst_from_text(txt)
            if ds and tbl:
                role = "lookup_ext" if re.search(r"\blookup_ext\s*\(", txt, flags=re.I) else "lookup"
                calls_by_fn[key].append({"role": role, "ds": ds, "sch": sch, "tbl": tbl})

    return calls_by_fn, fn_names

# ---------- main parse ----------
def parse_sap_ds(xml_path: str):
    root = ET.parse(xml_path).getroot()
    pm = build_parent_map(root)

    # project name (best effort)
    project_name = ""
    pj = root.find(".//DIProject")
    if pj is not None:
        project_name = clean(pj.attrib.get("name") or "")

    # collect all explicit job names (some exports repeat them)
    all_jobs = set()
    for n in root.iter():
        tg = strip_ns(n.tag)
        if tg == "dijob":
            nm = clean(n.attrib.get("name") or "")
            if nm: all_jobs.add(nm)
        elif tg == "diattribute" and (n.attrib.get("name") == "job_name"):
            val = clean(n.attrib.get("value") or "")
            if val: all_jobs.add(val)

    calls_by_fn, fn_names = index_user_functions(root)

    # rows keyed per DF/role/ds/sch/tbl
    pos_map = defaultdict(lambda: set())  # key -> set of positions (schemaOut>>column)
    count_map = defaultdict(int)
    header_rows = {}

    for df in root.iter():
        if strip_ns(df.tag) != "didataflow": continue
        df_name = clean(df.attrib.get("name") or "")

        # job name: nearest DIJob ancestor or fallback when there is only one job total
        job_name = ""
        cur = df
        for _ in range(60):
            cur = pm.get(cur)
            if cur is None: break
            if strip_ns(cur.tag) == "dijob":
                job_name = clean(cur.attrib.get("name") or "")
                if job_name: break
        if not job_name and len(all_jobs) == 1:
            job_name = list(all_jobs)[0]

        # make a placeholder row holder for this DF/job/project
        header_rows[(project_name, job_name, df_name)] = True

        # ---- A) direct FUNCTION_CALL lookups inside DF ----
        for fc in df.iter():
            if strip_ns(fc.tag) != "function_call": continue
            nm = low(fc.attrib.get("name"))
            if nm not in ("lookup","lookup_ext"): continue
            ds, sch, tbl = dst_from_function_call(fc)
            if not (ds and tbl):
                ds, sch, tbl = extract_dst_from_text("".join(fc.itertext()))
            if not (ds and tbl): continue

            schema_out = nearest_schema_out(fc, pm)
            col = nearest_output_column(fc, pm)  # treat all lookups as column-level
            role = "lookup_ext" if nm == "lookup_ext" else "lookup"

            key = (project_name, job_name, df_name, role, ds, sch, tbl)
            pos_map[key].add(f"{schema_out}>>{col if col else ''}")
            count_map[key] += 1

        # ---- B) text-based calls inside DF (ui_mapping_text / DIExpression) ----
        for e in df.iter():
            tg = strip_ns(e.tag)
            txt = ""
            if tg == "diattribute" and e.attrib.get("name") == "ui_mapping_text":
                txt = e.attrib.get("value","")
            elif tg == "diexpression":
                txt = e.attrib.get("expr","")
            else:
                continue
            txt_low = low(txt)

            # B1: function-by-name references to DIScriptFunction
            called = [fn for fn in fn_names if fn in txt_low]
            for fn in called:
                for call in calls_by_fn.get(fn, []):
                    ds, sch, tbl = call["ds"], call["sch"], call["tbl"]
                    role = call["role"]
                    schema_out = nearest_schema_out(e, pm)
                    col = nearest_output_column(e, pm)
                    key = (project_name, job_name, df_name, role, ds, sch, tbl)
                    pos_map[key].add(f"{schema_out}>>{col if col else ''}")
                    count_map[key] += 1

            # B2: direct lookup(...) / lookup_ext(...) in text
            if "lookup(" in txt_low or "lookup_ext(" in txt_low or "[" in txt_low:
                ds, sch, tbl = extract_dst_from_text(txt)
                if ds and tbl:
                    role = "lookup_ext" if "lookup_ext(" in txt_low else "lookup"
                    schema_out = nearest_schema_out(e, pm)
                    col = nearest_output_column(e, pm)
                    key = (project_name, job_name, df_name, role, ds, sch, tbl)
                    pos_map[key].add(f"{schema_out}>>{col if col else ''}")
                    count_map[key] += 1

    # ----- build final rows (distinct positions, comma-separated) -----
    out_rows = []
    for key, pos_set in pos_map.items():
        project_name, job_name, df_name, role, ds, sch, tbl = key
        # distinct, stable order
        positions = ", ".join(sorted(p for p in pos_set if p))
        out_rows.append({
            "project_name": project_name,
            "job_name": job_name,
            "dataflow_name": df_name,
            "role": role,
            "datastore": ds,
            "schema": sch,
            "table": tbl,
            "lookup_position": positions,
            "used_in_transform": role,
            "in_transform_used_count": len(pos_set)  # count distinct columns
        })

    # Also include sources/targets if you want (unchanged from your last good run),
    # but since this request is lookup-focused, we return only lookups here.
    # If you need sources/targets merged as before, ping me and I’ll add them back cleanly.

    df = pd.DataFrame(out_rows).sort_values(
        ["project_name","job_name","dataflow_name","role","datastore","schema","table"]
    )
    return df

# ---------- main ----------
def main():
    # Hardcode path here
    XML_PATH = r"C:\path\to\your\export.xml"   # <— change me
    df = parse_sap_ds(XML_PATH)
    out_xlsx = "xml_lineage_output.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        df.to_excel(xw, sheet_name="lineage", index=False)
    print(f"Wrote {out_xlsx} with {len(df)} rows")

if __name__ == "__main__":
    main()
