#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V9: SAP DS export XML lineage extractor

Keeps V8 behavior and:
- lookup: captured from UI mapping text (unchanged from V8)
- lookup_ext: ONLY from <FUNCTION_CALL name="lookup_ext" ...> (no DIExpression fallback)
- Custom SQL: STRICT like V8 — only when a real <sql_text> exists and is non-empty;
  also capture datastore from DIAttribute name="database_datastore"
"""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from typing import Dict, Iterable
import pandas as pd

def strip_ns(tag): 
    return re.sub(r"^\{.*\}", "", tag) if isinstance(tag, str) else ""

def lower(s): 
    return (s or "").strip().lower()

def attrs_ci(e) -> Dict[str, str]:
    return { (k or "").lower(): (v or "") for k, v in getattr(e, "attrib", {}).items() }

def build_parent_map(root): 
    return {c: p for p in root.iter() for c in p}

def ancestors(e, pm, lim=200) -> Iterable:
    cur = e
    for _ in range(lim):
        if cur is None: break
        yield cur
        cur = pm.get(cur)

def collect_text(n):
    parts=[]
    if hasattr(n,"attrib"):
        parts.extend([str(v) for v in n.attrib.values() if v])
    if n.text: 
        parts.append(n.text)
    for c in list(n):
        parts.append(collect_text(c))
        if c.tail: parts.append(c.tail)
    return " ".join([p for p in parts if p])

def dedupe(seq):
    out, seen = [], set()
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def canon(s:str)->str: 
    return re.sub(r'[^A-Z0-9]','',(s or '').upper())

def _strip_wrappers(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().strip('"').strip("'")
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        s = s[1:-1]
    return s

def _norm_key(s: str) -> str:
    s = _strip_wrappers(s)
    s = re.sub(r"[\{\}\[\]]", "", s)
    s = re.sub(r"\s+", " ", s).replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s.upper()

def _pretty(s: str) -> str:
    s = _strip_wrappers(s)
    s = re.sub(r"[\{\}\[\]]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class NameBag:
    def __init__(self): self.best = None
    def _score(self, s):
        if not s: return (-1, -1, -1)
        has_space = 1 if " " in s else 0
        clean     = 1 if (s == _pretty(s)) else 0
        return (has_space, clean, len(s))
    def add(self, s):
        s = _pretty(s)
        if not s: return
        if self.best is None or self._score(s) > self._score(self.best):
            self.best = s
    def get(self, fallback=""): return self.best or _pretty(fallback)

def nearest_attr_up(e, parent_map, name_lc: str):
    name_lc = name_lc.lower()
    for n in e.iter():
        if isinstance(n.tag, str) and strip_ns(n.tag).lower() == "diattribute":
            a = attrs_ci(n)
            if a.get("name","").lower() == name_lc:
                return (a.get("value") or "").strip()
    cur = e
    for _ in range(200):
        cur = parent_map.get(cur)
        if not cur: break
        for n in cur.iter():
            if isinstance(n.tag, str) and strip_ns(n.tag).lower() == "diattribute":
                a = attrs_ci(n)
                if a.get("name","").lower() == name_lc:
                    return (a.get("value") or "").strip()
    return ""

DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")
CALLSTEP_TAGS = ("dicallstep","callstep")

NAME_CHARS    = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"
DOT_NORMALIZE = re.compile(r"\s*\.\s*")
HAS_LOOKUP     = re.compile(r'\blookup(?!_)\s*\(', re.I)
LOOKUP_CALL_RE     = re.compile(rf'\blookup(?!_)\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)
LOOKUP_ARGS_RE     = re.compile(r'\blookup(?!_)\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*(?:,|\))', re.I|re.S)
LOOKUP_EXT_DOTTED_RE = re.compile(rf'\blookup_ext\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)

def _valid_triplet(ds: str, sch: str, tbl: str) -> bool:
    return bool(ds and tbl)

def extract_lookup_from_text(text: str):
    if not text: return ("","","")
    t = DOT_NORMALIZE.sub(".", text)
    m1 = LOOKUP_CALL_RE.search(t)
    if m1 and _valid_triplet(m1.group(1), m1.group(2), m1.group(3)):
        return m1.group(1).strip(), m1.group(2).strip(), m1.group(3).strip()
    m2 = LOOKUP_ARGS_RE.search(t)
    if m2 and _valid_triplet(m2.group(1), m2.group(2), m2.group(3)):
        return m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()
    return ("","","")

def extract_lookup_ext_from_function_call(node_text: str):
    t = DOT_NORMALIZE.sub(".", node_text or "")
    m1 = LOOKUP_EXT_DOTTED_RE.search(t)
    if m1 and _valid_triplet(m1.group(1), m1.group(2), m1.group(3)):
        return m1.group(1).strip(), m1.group(2).strip(), m1.group(3).strip()
    return ("","","")

Record = namedtuple("Record", [
    "project_name","job_name","dataflow_name",
    "role","datastore","schema","table",
    "transformation_position","transformation_usages_count","custom_sql_text"
])

SQL_TABLE_RE = re.compile(r'\b(?:from|join)\s+([A-Z0-9_\."]+)', re.I)

def parse_single_xml(xml_path: str) -> pd.DataFrame:
    tree=ET.parse(xml_path); root=tree.getroot()
    pm=build_parent_map(root)
    display_ds  = defaultdict(NameBag)
    display_sch = defaultdict(NameBag)
    display_tbl = defaultdict(NameBag)

    def remember_display(ds, sch, tbl):
        k=(_norm_key(ds),_norm_key(sch),_norm_key(tbl))
        display_ds[k].add(ds); display_sch[k].add(sch); display_tbl[k].add(tbl)

    lookup_pos    = defaultdict(list)
    lookup_ext_pos= defaultdict(set)
    rows_custom_sql = []

    for e in root.iter():
        tag=lower(strip_ns(e.tag)); a=attrs_ci(e)

        if tag=="diattribute" and lower(a.get("name",""))=="ui_mapping_text":
            txt=a.get("value") or e.text or ""
            if HAS_LOOKUP.search(txt):
                dsl,schl,tbl=extract_lookup_from_text(txt)
                if dsl and tbl:
                    remember_display(dsl,schl,tbl)
                    lookup_pos[(_norm_key(dsl),_norm_key(schl),_norm_key(tbl))].append("")

        if tag=="function_call" and lower(a.get("name",""))=="lookup_ext":
            dsx,schx,tbx = extract_lookup_ext_from_function_call(collect_text(e))
            if dsx and tbx:
                remember_display(dsx,schx,tbx)
                lookup_ext_pos[(_norm_key(dsx),_norm_key(schx),_norm_key(tbx))].add("")

        if tag == "ditransformcall" and (a.get("typeid") == "24" or lower(a.get("type") or "") == "sql"):
            sql_text = ""
            for ch in e.iter():
                if lower(strip_ns(ch.tag)) == "sql_text":
                    sql_text = (ch.text or "").strip()
                    break
            if not sql_text:
                continue
            ds_for_sql = nearest_attr_up(e, pm, "database_datastore") or ""
            tables = []
            for m in SQL_TABLE_RE.finditer(sql_text):
                val = m.group(1).strip().strip('"')
                if val: tables.append(val)
            schemas = ", ".join(sorted(dedupe([t.split(".")[0] if "." in t else t for t in tables]))) if tables else "CUSTOM_SQL"
            tablenames = ", ".join(sorted(dedupe([t.split(".")[-1] for t in tables]))) if tables else "SQL"
            safe_sql_text = "\"" + " ".join(sql_text.split()).replace('"', '""') + "\""
            rows_custom_sql.append(("", "", "", "source", ds_for_sql, schemas, tablenames, "", len(tables) or 1, safe_sql_text))

    rows=[]
    for (dsN,schN,tblN), positions in lookup_pos.items():
        dsD,schD,tblD=display_ds[(dsN,schN,tblN)].get(dsN), display_sch[(dsN,schN,tblN)].get(schN), display_tbl[(dsN,schN,tblN)].get(tblN)
        rows.append(("", "", "", "lookup", dsD, schD, tblD, "", len(positions), ""))

    for (dsN,schN,tblN), posset in lookup_ext_pos.items():
        dsD,schD,tblD=display_ds[(dsN,schN,tblN)].get(dsN), display_sch[(dsN,schN,tblN)].get(schN), display_tbl[(dsN,schN,tblN)].get(tblN)
        rows.append(("", "", "", "lookup_ext", dsD, schD, tblD, "", len(posset), ""))

    for r in rows_custom_sql:
        rows.append(Record(*r))

    return pd.DataFrame(rows)

def main():
    xml_path  = r"C:/Users/raksahu/Downloads/python/input/export_df.xml"
    out_xlsx  = r"C:/Users/raksahu/Downloads/python/input/output_v9.xlsx"
    df = parse_single_xml(xml_path)
    df.to_excel(out_xlsx, index=False)
    print(f"Done: {len(df)} rows written to {out_xlsx}")

if __name__ == "__main__":
    main()
