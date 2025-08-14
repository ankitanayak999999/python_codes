#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAP Data Services XML parser — V8
- Based on your V7, with the missing V6 feature restored:
  * Global object registry so references inside <DATAFLOW> resolve to objects defined outside
- Keeps V7's Custom SQL extraction
- Single-input usage preserved (positional or --in), optional --out
"""

import argparse
import os
import sys
import datetime
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Optional

import pandas as pd  # plain import (no try/except)

# =========================
# Small helpers
# =========================

def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def norm(s: Optional[str]) -> str:
    return (s or "").strip()

def text_of(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()

def first_nonempty(*vals) -> str:
    for v in vals:
        v = norm(v)
        if v:
            return v
    return ""

def looks_like_sql(s: str) -> bool:
    if not s:
        return False
    sl = s.lower()
    return any(w in sl for w in [
        "select ", " insert ", " update ", " delete ", " merge ", " from ", " create ", " with "
    ]) or sl.startswith("select ") or sl.startswith("with ")

# Attribute/tag candidates seen in SAP DS exports
ID_ATTRS   = ["id", "ID", "Id", "object_id", "objectId"]
NAME_ATTRS = ["name", "Name", "NAME", "object_name", "display_name", "label"]

REF_ATTRS  = [
    "ref", "ref_id", "refid", "object_ref", "objectRef",
    "source_ref", "target_ref", "table_ref", "function_ref",
    "transform_ref", "src_ref", "dst_ref", "refId"
]

SQL_TAG_HINTS = [
    "CUSTOM", "custom", "custom_sql", "sql", "sqlText",
    "sql_text", "SQL", "Sql", "text", "expression"
]
SQL_ATTR_HINTS = ["sql", "SQL", "text", "expression", "value", "string"]

DATAFLOW_HINTS = ["DATAFLOW", "Dataflow", "DATA_FLOW", "DF", "DIAGRAM", "diagram", "Graph"]
LINK_HINTS     = ["LINK", "Link", "EDGE", "edge", "Connection", "CONNECTOR", "connector"]

def is_dataflow_like(elem: ET.Element) -> bool:
    t = elem.tag.lower()
    return any(h.lower() in t for h in DATAFLOW_HINTS)

def is_link_like(elem: ET.Element) -> bool:
    t = elem.tag.lower()
    return any(h.lower() in t for h in LINK_HINTS)

def candidate_ref_value(elem: ET.Element) -> str:
    for k in REF_ATTRS:
        if k in elem.attrib and norm(elem.attrib[k]):
            return elem.attrib[k].strip()
    return ""

# =========================
# V6-style Global Object Registry (restored)
# =========================

class ObjectRegistry:
    """
    Collect ALL elements that have an ID (regardless of where they appear in the XML).
    This allows resolving references used inside <DATAFLOW> to objects defined outside.
    """
    def __init__(self):
        self.by_id: Dict[str, ET.Element] = {}
        self.names: Dict[str, List[str]] = defaultdict(list)
        self.types: Dict[str, str] = {}

    def add(self, elem: ET.Element):
        oid = self._get_id(elem)
        if not oid:
            return
        self.by_id[oid] = elem
        self.types[oid] = elem.tag
        nm = self._get_name(elem)
        if nm and nm not in self.names[oid]:
            self.names[oid].append(nm)

    def _get_id(self, elem: ET.Element) -> str:
        for k in ID_ATTRS:
            if k in elem.attrib:
                return elem.attrib[k]
        return ""

    def _get_name(self, elem: ET.Element) -> str:
        # 1) name as attribute
        for k in NAME_ATTRS:
            v = elem.attrib.get(k)
            if v and str(v).strip():
                return str(v).strip()
        # 2) <...NAME> child text
        for ch in elem.iter():
            tg = ch.tag.upper()
            if tg.endswith("NAME"):
                t = "".join(ch.itertext()).strip()
                if t:
                    return t
        return ""

    def resolve_name(self, oid: str) -> str:
        if not oid:
            return ""
        arr = self.names.get(oid, [])
        return arr[0] if arr else ""

def build_global_registry_from_root(root: ET.Element) -> ObjectRegistry:
    reg = ObjectRegistry()
    for elem in root.iter():
        if any(k in elem.attrib for k in ID_ATTRS):
            reg.add(elem)
    return reg

# =========================
# Custom SQL extraction (kept from V7)
# =========================

def sniff_sql_from_element(elem: ET.Element) -> List[str]:
    sqls: List[str] = []

    # 1) direct text
    t = text_of(elem)
    if looks_like_sql(t):
        sqls.append(t)

    # 2) attributes holding SQL/text
    for k, v in elem.attrib.items():
        if any(h.lower() in k.lower() for h in SQL_ATTR_HINTS):
            if looks_like_sql(v):
                sqls.append(v)

    # 3) child tags often carrying SQL
    for child in elem:
        tagl = child.tag.lower()
        if any(h.lower() in tagl for h in SQL_TAG_HINTS):
            tv = text_of(child)
            if looks_like_sql(tv):
                sqls.append(tv)

    # uniq
    out, seen = [], set()
    for s in sqls:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

# =========================
# Extraction (links + SQL), with global resolution restored
# =========================

Link = namedtuple("Link", [
    "dataflow_id", "dataflow_name",
    "from_id", "from_name",
    "to_id", "to_name",
    "edge_tag"
])

def collect_objects_table(reg: ObjectRegistry) -> pd.DataFrame:
    rows = []
    for oid, elem in reg.by_id.items():
        rows.append({
            "object_id": oid,
            "object_tag": elem.tag,
            "object_name": reg.resolve_name(oid),
        })
    df = pd.DataFrame(rows).sort_values(["object_tag", "object_name", "object_id"])
    return df.reset_index(drop=True)

def extract_links_and_sql(root: ET.Element, reg: ObjectRegistry) -> Tuple[pd.DataFrame, pd.DataFrame]:
    links: List[Link] = []
    sql_rows: List[Dict[str, str]] = []

    for df in root.iter():
        if not is_dataflow_like(df):
            continue

        # Identify dataflow
        df_id = first_nonempty(*[df.attrib.get(k, "") for k in ID_ATTRS])
        df_name = first_nonempty(*[df.attrib.get(k, "") for k in NAME_ATTRS])
        if not df_name and df_id:
            df_name = reg.resolve_name(df_id)

        # 1) Custom SQL inside this DF
        for sub in df.iter():
            for sql in sniff_sql_from_element(sub):
                sql_rows.append({
                    "dataflow_id": df_id,
                    "dataflow_name": df_name,
                    "object_tag": sub.tag,
                    "object_id": first_nonempty(*[sub.attrib.get(k, "") for k in ID_ATTRS]),
                    "object_name": first_nonempty(*[sub.attrib.get(k, "") for k in NAME_ATTRS]),
                    "sql": sql
                })

        # 2) Links (resolve endpoints even if defined outside DF)
        for sub in df.iter():
            if not is_link_like(sub):
                continue

            # attribute forms
            src = ""
            tgt = ""
            for cand in ["from", "from_ref", "src", "src_ref", "source", "source_ref"]:
                if cand in sub.attrib and norm(sub.attrib[cand]):
                    src = sub.attrib[cand].strip()
                    break
            for cand in ["to", "to_ref", "dst", "dst_ref", "target", "target_ref"]:
                if cand in sub.attrib and norm(sub.attrib[cand]):
                    tgt = sub.attrib[cand].strip()
                    break

            # child forms: <From ref=".."/><To ref=".."/>
            if not src or not tgt:
                child_src = None
                child_tgt = None
                for ch in sub:
                    tl = ch.tag.lower()
                    if "from" in tl or "src" in tl or "source" in tl:
                        rv = candidate_ref_value(ch)
                        if rv:
                            child_src = rv
                    if "to" in tl or "dst" in tl or "target" in tl:
                        rv = candidate_ref_value(ch)
                        if rv:
                            child_tgt = rv
                src = src or (child_src or "")
                tgt = tgt or (child_tgt or "")

            if src or tgt:
                src_name = reg.resolve_name(src) if src else ""
                tgt_name = reg.resolve_name(tgt) if tgt else ""
                links.append(Link(
                    dataflow_id=df_id,
                    dataflow_name=df_name,
                    from_id=src,
                    from_name=src_name,
                    to_id=tgt,
                    to_name=tgt_name,
                    edge_tag=sub.tag
                ))

    links_df = pd.DataFrame([l._asdict() for l in links]).sort_values(
        ["dataflow_name", "from_name", "to_name", "edge_tag"]
    ).reset_index(drop=True)

    sql_df = pd.DataFrame(sql_rows).sort_values(
        ["dataflow_name", "object_tag", "object_name", "object_id"]
    ).reset_index(drop=True)

    return links_df, sql_df

# =========================
# IO (single-file usage preserved)
# =========================

def parse_args():
    ap = argparse.ArgumentParser(description="SAP DS XML (V8: V7 + restored V6 object resolution)")
    # Support your existing style: either positional or --in
    ap.add_argument("positional_input", nargs="?", help="XML input file (optional positional).")
    ap.add_argument("--in", dest="input_path", help="XML input file path.")
    ap.add_argument("--out", dest="out", default=None, help="Output Excel file path (optional).")
    return ap.parse_args()

def resolve_input_path(args) -> str:
    path = args.input_path or args.positional_input
    if not path:
        print("[ERROR] Please provide the XML input file (positional or --in).")
        sys.exit(2)
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(2)
    return path

def default_out_path(in_path: str, user_out: Optional[str]) -> str:
    if user_out:
        return user_out
    base = os.path.splitext(os.path.basename(in_path))[0]
    return f"{base}_extract_{now_stamp()}.xlsx"

def write_excel(objects_df: pd.DataFrame, links_df: pd.DataFrame, sql_df: pd.DataFrame, out_xlsx: str):
    out_xlsx = os.path.abspath(out_xlsx)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        # Keep sheet names simple like V7 (objects/links/custom_sql)
        objects_df.to_excel(xw, index=False, sheet_name="objects")
        links_df.to_excel(xw, index=False, sheet_name="links")
        sql_df.to_excel(xw, index=False, sheet_name="custom_sql")
    print(f"[OK] Excel written: {out_xlsx}")

# =========================
# Main
# =========================

def main():
    args = parse_args()
    in_path = resolve_input_path(args)

    # Parse XML
    try:
        tree = ET.parse(in_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse XML: {e}")
        sys.exit(2)

    # Restore V6 behavior: build global registry (so external objects referenced inside DF resolve)
    print("[INFO] Building global registry (V6 behavior restored)...")
    reg = build_global_registry_from_root(root)
    print(f"[INFO] Registry objects: {len(reg.by_id)}")

    # Objects table (optional but useful; kept simple)
    print("[INFO] Collecting objects table...")
    objects_df = collect_objects_table(reg)

    # Links + Custom SQL (V7 behavior preserved)
    print("[INFO] Extracting links and custom SQL...")
    links_df, sql_df = extract_links_and_sql(root, reg)

    # Output
    out_path = default_out_path(in_path, args.out)
    print("[INFO] Writing outputs...")
    write_excel(objects_df, links_df, sql_df, out_path)

if __name__ == "__main__":
    main()
