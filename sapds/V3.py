#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# all good except scheam name for custum SQL

import re
from collections import defaultdict
from xml.etree import ElementTree as ET
from pathlib import Path
import pandas as pd

# ============ CONFIG ============
XML_PATH = r"C:\path\to\your\export.xml"  # Change to your XML path
OUT_XLSX = "xml_lineage_output.xlsx"
OUT_SHEET = "lineage"
# =================================

# ---------- helpers ----------
def t(tag):
    if not isinstance(tag, str): return ""
    return tag.split("}", 1)[1].lower() if tag.startswith("{") else tag.lower()

def low(s):
    return (s or "").lower()

def clean_txt(s):
    s = (s or "").strip().strip('"').strip("'")
    if len(s) >= 2 and s[0] in "[{" and s[-1] in "]}":
        s = s[1:-1]
    s = s.replace("[", "").replace("]", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def build_parent_map(root):
    pm = {}
    for p in root.iter():
        for c in list(p):
            pm[c] = p
    return pm

def find_ancestor(elem, names, pm):
    wanted = {n.lower() for n in names}
    cur = elem
    for _ in range(200):
        if cur is None: break
        if t(cur.tag) in wanted:
            return cur
        cur = pm.get(cur)
    return None

def schema_out_of(elem, pm):
    node = find_ancestor(elem, ["DISchema"], pm)
    if node is not None:
        nm = clean_txt(node.attrib.get("name") or node.attrib.get("displayName") or "")
        return nm or "Join"
    return "Join"

def outcol_of(elem, pm):
    node = find_ancestor(elem, ["DIElement"], pm)
    if node is not None:
        nm = clean_txt(node.attrib.get("name") or node.attrib.get("displayName") or "")
        if nm:
            return nm
    return ""

# ---------- DS/Schema/Table from expressions ----------
RGX_TRIPLE  = re.compile(r"\b(?:lookup|lookup[_ ]?ext)\s*\(\s*([^\s,()\[\]]+)\s*,\s*([^\s,()\[\]]+)\s*,\s*([^\s,()\[\]]+)", re.I)
RGX_BRACKET = re.compile(r"\[\s*([^\s\].]+)\.([^\s\].]+)\.([^\s\].]+)\s*\]", re.I)

def dst_from_text(txt):
    if not txt: return ("", "", "")
    m = RGX_TRIPLE.search(txt)
    if m: return clean_txt(m.group(1)), clean_txt(m.group(2)), clean_txt(m.group(3))
    m = RGX_BRACKET.search(txt)
    if m: return clean_txt(m.group(1)), clean_txt(m.group(2)), clean_txt(m.group(3))
    return ("", "", "")

# ---------- main parser ----------
def parse_xml(path):
    root = ET.parse(path).getroot()
    pm = build_parent_map(root)
    rows = []

    # project name
    proj_node = root.find(".//DIProject")
    project_name = proj_node.attrib.get("name") if proj_node is not None else ""

    # sources
    for node in root.findall(".//DIInputTable"):
        job_name = ""
        df_name = ""
        ds = clean_txt(node.attrib.get("datastoreName", ""))
        schema = clean_txt(node.attrib.get("schemaName", ""))
        table = clean_txt(node.attrib.get("name", ""))
        rows.append([project_name, job_name, df_name, "source", ds, schema, table, "", 0])

    # targets
    for node in root.findall(".//DIOutputTable"):
        job_name = ""
        df_name = ""
        ds = clean_txt(node.attrib.get("datastoreName", ""))
        schema = clean_txt(node.attrib.get("schemaName", ""))
        table = clean_txt(node.attrib.get("name", ""))
        rows.append([project_name, job_name, df_name, "target", ds, schema, table, "", 0])

    # lookups & lookup_ext
    for elem in root.findall(".//DIExpression"):
        text = elem.text or ""
        if re.search(r"\blookup\b", text, re.I) or re.search(r"\blookup[_ ]?ext\b", text, re.I):
            ds, schema, table = dst_from_text(text)
            role = "lookup_ext" if "lookup_ext" in text.lower() else "lookup"
            lookup_pos = f"{schema_out_of(elem, pm)}>>{outcol_of(elem, pm)}"
            rows.append([project_name, "", "", role, ds, schema, table, lookup_pos, 1])

    # deduplicate lookup positions
    df = pd.DataFrame(rows, columns=[
        "project_name", "job_name", "dataflow_name", "role",
        "datastore", "schema", "table", "lookup_position", "lp_used_count"
    ])
    df = (df.groupby(["project_name", "job_name", "dataflow_name", "role", "datastore", "schema", "table"])
            .agg({"lookup_position": lambda x: ",".join(sorted(set(v for v in x if v))),
                  "lp_used_count": "sum"}).reset_index())
    return df

# ---------- run ----------
df = parse_xml(XML_PATH)
df.to_excel(OUT_XLSX, sheet_name=OUT_SHEET, index=False)
print(f"Saved to {OUT_XLSX}")
