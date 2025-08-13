import re
import sys
from collections import defaultdict
from xml.etree import ElementTree as ET

import pandas as pd

# ---------------------------- helpers ----------------------------

def strip_ns(tag: str) -> str:
    if tag and tag[0] == "{":
        return tag.split("}", 1)[1].lower()
    return (tag or "").lower()

def lower(s): return (s or "").lower()

def clean(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    # remove leading/trailing brackets, collapse spaces
    s = re.sub(r"^\[|\]$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def norm(s: str) -> str:
    return clean(s).lower()

def gather_text(elem) -> str:
    # concatenates text + tail from subtree
    txts = []
    if elem.text:
        txts.append(elem.text)
    for e in elem.iter():
        if e is elem: 
            continue
        if e.text:
            txts.append(e.text)
        if e.tail:
            txts.append(e.tail)
    if elem.tail:
        txts.append(elem.tail)
    return " ".join(txts)

def build_parent_map(root):
    pm = {}
    for p in root.iter():
        for c in p:
            pm[c] = p
    return pm

def nearest_schema(elem, parent_map):
    # walk up until a DISchema appears; take its name, else "Join"
    cur = elem
    for _ in range(50):
        if cur is None: break
        if strip_ns(cur.tag) == "dischema":
            nm = (cur.attrib.get("name") or "").strip()
            return nm or "Join"
        cur = parent_map.get(cur)
    return "Join"

def nearest_element(elem, parent_map):
    # walk up to find the output column element (<DIElement name="...">)
    cur = elem
    for _ in range(50):
        if cur is None: break
        if strip_ns(cur.tag) == "dielement":
            nm = (cur.attrib.get("name") or "").strip()
            if nm:
                return nm
        cur = parent_map.get(cur)
    return None

# text patterns for in-expression lookups
RGX_LU = re.compile(r"\blookup\s*\(", re.I)
RGX_LUX = re.compile(r"\blookup_ext\s*\(", re.I)

# capture DS, schema, table from textual calls:
#   lookup( DS_ODS, BRIOVIEW, BANK_CMMT_TYPE_GL_XREF , ...
RGX_TRIPLE = re.compile(
    r"\b(?:lookup|lookup_ext)\s*\(\s*([^\s,()\[\]]+)\s*,\s*([^\s,()\[\]]+)\s*,\s*([^\s,()\[\]]+)",
    re.I,
)

# alternative format like [DS_ODS.BRIOVIEW.BANK_CMMT_TYPE_GL_XREF]
RGX_BRACKET_TRIPLE = re.compile(
    r"\[\s*([^\s\].]+)\.([^\s\].]+)\.([^\s\].]+)\s*\]",
    re.I,
)

def extract_triple_from_text(txt: str):
    m = RGX_TRIPLE.search(txt)
    if m:
        return clean(m.group(1)), clean(m.group(2)), clean(m.group(3))
    m = RGX_BRACKET_TRIPLE.search(txt)
    if m:
        return clean(m.group(1)), clean(m.group(2)), clean(m.group(3))
    return None, None, None

# --------------------- function index (DIScriptFunction) ---------------------

def index_script_functions(root):
    """
    Returns:
      fn_calls: { FN_NAME_UPPER: [(is_ext, ds, sch, tbl), ...] }
    We read all <DIScriptFunction> bodies for embedded <FUNCTION_CALL name="lookup(_ext)">.
    """
    fn_calls = defaultdict(list)
    for fn in root.iter():
        if strip_ns(fn.tag) != "discriptfunction":
            continue
        fn_name = clean(fn.attrib.get("name", ""))
        if not fn_name:
            continue
        fname_u = fn_name.upper()

        # direct FUNCTION_CALL children inside the function
        for fc in fn.iter():
            if strip_ns(fc.tag) != "function_call":
                continue
            call_name = lower(fc.attrib.get("name"))
            if call_name not in ("lookup", "lookup_ext"):
                continue
            is_ext = (call_name == "lookup_ext")
            ds  = clean(fc.attrib.get("tabledatastore"))
            sch = clean(fc.attrib.get("tableowner"))
            tbl = clean(fc.attrib.get("tablename"))
            if not (ds and tbl):
                # try to recover from inline text (rare in functions, but safe)
                t = gather_text(fc)
                ds2, sch2, tbl2 = extract_triple_from_text(t)
                ds  = ds  or ds2
                sch = sch or sch2
                tbl = tbl or tbl2
            if ds and tbl:
                fn_calls[fname_u].append((is_ext, ds, sch, tbl))

        # also scan the whole function text for raw lookup/lookup_ext in expressions
        blob = gather_text(fn)
        for pat, is_ext in ((RGX_LU, False), (RGX_LUX, True)):
            if pat.search(blob):
                ds, sch, tbl = extract_triple_from_text(blob)
                if ds and tbl:
                    fn_calls[fname_u].append((is_ext, ds, sch, tbl))

    return fn_calls

# ----------------------------- main parser -----------------------------------

def parse_xml(xml_path: str):
    root = ET.parse(xml_path).getroot()
    pm   = build_parent_map(root)

    # project & job maps (best-effort)
    project_name = ""
    pj = next((n for n in root.iter() if strip_ns(n.tag) == "diproject"), None)
    if pj:
        project_name = clean(pj.attrib.get("name"))

    # Map DF name -> job name (via nearest DIJob ancestor, fallback by attribute)
    df_to_job = {}
    for df in root.iter():
        if strip_ns(df.tag) != "didataflow":
            continue
        df_name = clean(df.attrib.get("name"))
        # walk up to find job
        job = None
        cur = pm.get(df)
        for _ in range(50):
            if cur is None: break
            if strip_ns(cur.tag) == "dijob":
                job = cur
                break
            cur = pm.get(cur)
        job_name = ""
        if job:
            job_name = clean(job.attrib.get("name") or "")
            # explicit attribute sometimes present
            for a in job.iter():
                if strip_ns(a.tag) == "diattribute" and lower(a.attrib.get("name")) == "job_name":
                    job_name = clean(a.attrib.get("value") or job_name) or job_name
                    break
        df_to_job[df_name] = job_name

    # index custom script functions with embedded lookups
    fn_idx = index_script_functions(root)

    # per-DF collectors
    rows = []

    # iterate each dataflow
    for df in root.iter():
        if strip_ns(df.tag) != "didataflow":
            continue
        df_name = clean(df.attrib.get("name"))

        # de-dup within a DF
        seen_src = set()
        seen_tgt = set()
        seen_lookup = set()     # (role, ds, sch, tbl, schema_out, col)
        seen_lookup_ext = set() # (role, ds, sch, tbl, schema_out)

        # 1) sources/targets from tables inside DF
        for t in df.iter():
            tg = strip_ns(t.tag)
            if tg in ("didatabasetablesource", "ditablesource"):
                ds  = clean(t.attrib.get("datastorename") or t.attrib.get("datastoreName"))
                sch = clean(t.attrib.get("ownername") or t.attrib.get("ownerName"))
                tbl = clean(t.attrib.get("tablename") or t.attrib.get("tableName"))
                sig = (norm(ds), norm(sch), norm(tbl))
                if sig not in seen_src:
                    seen_src.add(sig)
                    rows.append((project_name, df_to_job.get(df_name,""), df_name,
                                 "source", ds, sch, tbl, "", "source", 0))
            elif tg in ("didatabasetabletarget", "ditabletarget"):
                ds  = clean(t.attrib.get("datastorename") or t.attrib.get("datastoreName"))
                sch = clean(t.attrib.get("ownername") or t.attrib.get("ownerName"))
                tbl = clean(t.attrib.get("tablename") or t.attrib.get("tableName"))
                sig = (norm(ds), norm(sch), norm(tbl))
                if sig not in seen_tgt:
                    seen_tgt.add(sig)
                    rows.append((project_name, df_to_job.get(df_name,""), df_name,
                                 "target", ds, sch, tbl, "", "target", 0))

        # 2) direct FUNCTION_CALL lookup / lookup_ext inside DF
        for fc in df.iter():
            if strip_ns(fc.tag) != "function_call":
                continue
            call = lower(fc.attrib.get("name"))
            if call not in ("lookup", "lookup_ext"):
                continue
            ds  = clean(fc.attrib.get("tabledatastore"))
            sch = clean(fc.attrib.get("tableowner"))
            tbl = clean(fc.attrib.get("tablename"))
            schema_out = nearest_schema(fc, pm)
            col = nearest_element(fc, pm)

            # fallback to text if attributes missing
            if not (ds and tbl):
                blob = gather_text(fc)
                ds2, sch2, tbl2 = extract_triple_from_text(blob)
                ds, sch, tbl = ds or ds2, sch or sch2, tbl or tbl2

            if not (ds and tbl):
                continue

            if call == "lookup_ext":
                sig = ("lookup_ext", norm(ds), norm(sch), norm(tbl), norm(schema_out))
                if sig not in seen_lookup_ext:
                    seen_lookup_ext.add(sig)
                    rows.append((project_name, df_to_job.get(df_name,""), df_name,
                                 "lookup_ext", ds, sch, tbl, schema_out, "lookup_ext", 0))
            else:
                # column-level lookup
                sig = ("lookup", norm(ds), norm(sch), norm(tbl), norm(schema_out), norm(col or ""))
                if sig not in seen_lookup:
                    seen_lookup.add(sig)
                    pos = f"{schema_out}>>{col}" if col else schema_out
                    rows.append((project_name, df_to_job.get(df_name,""), df_name,
                                 "lookup", ds, sch, tbl, pos, "lookup", 0))

        # 3) DIExpression / ui_mapping_text inside DF (raw text calls or custom function names)
        for e in df.iter():
            tg = strip_ns(e.tag)
            if tg == "diattribute" and lower(e.attrib.get("name")) == "ui_mapping_text":
                txt = (e.attrib.get("value") or "").strip()
            elif tg == "diexpression":
                txt = gather_text(e)
            else:
                continue

            if not txt:
                continue

            schema_out = nearest_schema(e, pm)
            col        = nearest_element(e, pm)

            # direct textual lookup()
            if RGX_LU.search(txt):
                ds, sch, tbl = extract_triple_from_text(txt)
                if ds and tbl:
                    sig = ("lookup", norm(ds), norm(sch), norm(tbl), norm(schema_out), norm(col or ""))
                    if sig not in seen_lookup:
                        seen_lookup.add(sig)
                        pos = f"{schema_out}>>{col}" if col else schema_out
                        rows.append((project_name, df_to_job.get(df_name,""), df_name,
                                     "lookup", ds, sch, tbl, pos, "lookup", 0))

            # direct textual lookup_ext()
            if RGX_LUX.search(txt):
                ds, sch, tbl = extract_triple_from_text(txt)
                if ds and tbl:
                    sig = ("lookup_ext", norm(ds), norm(sch), norm(tbl), norm(schema_out))
                    if sig not in seen_lookup_ext:
                        seen_lookup_ext.add(sig)
                        rows.append((project_name, df_to_job.get(df_name,""), df_name,
                                     "lookup_ext", ds, sch, tbl, schema_out, "lookup_ext", 0))

            # custom function calls -> expand using fn_idx
            called = set()
            for fname_u in fn_idx.keys():
                if re.search(rf"\b{re.escape(fname_u)}\s*\(", txt, re.I):
                    called.add(fname_u)
            for fname_u in called:
                for is_ext, ds, sch, tbl in fn_idx[fname_u]:
                    if is_ext:
                        sig = ("lookup_ext", norm(ds), norm(sch), norm(tbl), norm(schema_out))
                        if sig not in seen_lookup_ext:
                            seen_lookup_ext.add(sig)
                            rows.append((project_name, df_to_job.get(df_name,""), df_name,
                                         "lookup_ext", ds, sch, tbl, schema_out, "lookup_ext", 0))
                    else:
                        sig = ("lookup", norm(ds), norm(sch), norm(tbl), norm(schema_out), norm(col or ""))
                        if sig not in seen_lookup:
                            seen_lookup.add(sig)
                            pos = f"{schema_out}>>{col}" if col else schema_out
                            rows.append((project_name, df_to_job.get(df_name,""), df_name,
                                         "lookup", ds, sch, tbl, pos, "lookup", 0))

        # 4) Count positions per (DF, role, DS, SCH, TBL)
        # aggregate after DF done
        # (we’ll compute counts globally below)

    # materialize DataFrame
    df = pd.DataFrame(rows, columns=[
        "project_name", "job_name", "dataflow_name",
        "role", "datastore", "schema", "table",
        "lookup_position", "used_in_transform", "in_transform_used_count"
    ])

    # clean again & drop dupes (same role/ds/sch/tbl/pos)
    for c in ("datastore", "schema", "table", "lookup_position", "project_name", "job_name", "dataflow_name"):
        df[c] = df[c].map(clean)

    # consolidate multiple positions into comma-separated & count unique positions
    key_cols = ["project_name", "job_name", "dataflow_name", "role", "datastore", "schema", "table", "used_in_transform"]
    def agg_pos(g):
        pos = [p for p in g["lookup_position"].tolist() if p]
        uniq = []
        seen = set()
        for p in pos:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return pd.Series({
            "lookup_position": ", ".join(uniq),
            "in_transform_used_count": len(uniq)
        })
    if not df.empty:
        df = df.groupby(key_cols, as_index=False).apply(agg_pos)

    # order like your template
    df = df[[
        "project_name", "job_name", "dataflow_name",
        "role", "datastore", "schema", "table",
        "lookup_position", "used_in_transform", "in_transform_used_count"
    ]].sort_values(["dataflow_name","role","schema","table"]).reset_index(drop=True)

    return df

# ------------------------------ main ----------------------------------------

def main():
    # >>>>>>>>>>>> SET YOUR XML PATH HERE <<<<<<<<<<<<
    XML_PATH = r"PATH\TO\YOUR\export.xml"  # <-- hardcode here

    out_xlsx = "xml_lineage_output.xlsx"

    df = parse_xml(XML_PATH)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        # One sheet with everything; you can filter in Excel
        df.to_excel(xw, sheet_name="lineage", index=False)

    print(f"Done. Wrote: {out_xlsx}")
    # If you also want a quick per‑DF sheet split, uncomment below:
    # with pd.ExcelWriter("xml_lineage_by_df.xlsx", engine="openpyxl") as xw:
    #     for df_name, g in df.groupby("dataflow_name"):
    #         safe = re.sub(r"[\\/*?:\[\]]", "_", df_name)[:31] or "sheet"
    #         g.to_excel(xw, sheet_name=safe, index=False)

if __name__ == "__main__":
    main()
