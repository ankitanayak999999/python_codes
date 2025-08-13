#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import xml.etree.ElementTree as ET
from collections import defaultdict, deque, namedtuple
import pandas as pd
from pathlib import Path

# --------- config ---------
XML_PATH = r"C:\path\to\your\export.xml"
OUT_CSV  = r"export_xml_lineage.csv"

# --------- small utils ---------
def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag) if isinstance(tag, str) else ""
def lower(s): return (s or "").strip().lower()
def clean(s):
    if s is None: return ""
    s = str(s).strip().strip('"').strip("'")
    if s.startswith("[") and s.endswith("]"): s = s[1:-1]
    if s.startswith("{") and s.endswith("}"): s = s[1:-1]
    s = s.replace("[","").replace("]","")
    s = re.sub(r"\s+", " ", s).strip()
    return s
def norm(s): return re.sub(r"[^A-Z0-9]", "", clean(s).upper())
def dedupe_keep_order(vals):
    out, seen = [], set()
    for v in vals:
        if not v: continue
        if v not in seen:
            seen.add(v); out.append(v)
    return out
def gather_text(node):
    parts=[]
    if hasattr(node,"attrib"): parts += [str(v) for v in node.attrib.values() if v]
    if node.text: parts.append(node.text)
    for c in list(node):
        parts.append(gather_text(c))
        if c.tail: parts.append(c.tail)
    return " ".join([p for p in parts if p])

# --------- patterns ---------
NAME_CHARS  = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"
DOTS        = re.compile(r"\s*\.\s*")
HAS_LOOKUP      = re.compile(r'\blookup(?!_)\s*\(', re.I)
HAS_LOOKUP_EXT  = re.compile(r'\blookup[_ ]?ext\s*\(', re.I)
LOOKUP_ARGS     = re.compile(r'\blookup(?!_)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,\)]+)', re.I|re.S)
LOOKUP_DOTTED   = re.compile(rf'\blookup(?!_)\s*\(\s*"?({NAME_CHARS})"?\s*\.\s*"?({NAME_CHARS})"?\s*\.\s*"?({NAME_CHARS})"?', re.I)
LOOKUP_EXT_ARGS = re.compile(r'\blookup[_ ]?ext\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,\)]+)', re.I|re.S)
LOOKUP_EXT_DOTTED = re.compile(rf'\blookup[_ ]?ext\s*\(\s*"?({NAME_CHARS})"?\s*\.\s*"?({NAME_CHARS})"?\s*\.\s*"?({NAME_CHARS})"?', re.I)
LOOKUP_EXT_KV = re.compile(
    r'\blookup[_ ]?ext\s*\([^)]*?'
    r'(?:tableDatastore|tabledatastore|tabledatastorename|datastorename)\s*=\s*([\'"]?)(?P<ds>[^\'",)\s]+)\1[^)]*?'
    r'(?:tableOwner|tableowner|ownername)\s*=\s*([\'"]?)(?P<own>[^\'",)\s]+)\3[^)]*?'
    r'(?:tableName|tablename|name)\s*=\s*([\'"]?)(?P<tbl>[^\'",)\s]+)\5',
    re.I|re.S
)

CACHE_WORDS  = {"PRE_LOAD_CACHE","POST_LOAD_CACHE","CACHE","PRE_LOAD","POST_LOAD"}
POLICY_WORDS = {"MAX","MIN","MAX-NS","MIN-NS","MAX_NS","MIN_NS"}
def is_meta(t):
    t=(t or "").strip().strip('"').strip("'").upper().replace(" ","_")
    return t in CACHE_WORDS or t in POLICY_WORDS
def looks_table(t):
    t=clean(t)
    if not t or t.isdigit() or is_meta(t): return False
    return bool(re.search(r"[A-Za-z]", t)) and len(t)>=3
def valid_trip(ds, sch, tbl): return bool(ds) and looks_table(tbl) and not is_meta(sch)

def extract_triple(text, is_ext):
    if not text: return ("","","")
    t = DOTS.sub(".", text)

    if is_ext:
        m = LOOKUP_EXT_KV.search(t)
        if m and valid_trip(m.group("ds"), m.group("own"), m.group("tbl")):
            return clean(m.group("ds")), clean(m.group("own")), clean(m.group("tbl"))

    m = (LOOKUP_EXT_DOTTED if is_ext else LOOKUP_DOTTED).search(t)
    if m and valid_trip(m.group(1), m.group(2), m.group(3)):
        return clean(m.group(1)), clean(m.group(2)), clean(m.group(3))

    m = (LOOKUP_EXT_ARGS if is_ext else LOOKUP_ARGS).search(t)
    if m and valid_trip(m.group(1), m.group(2), m.group(3)):
        return clean(m.group(1)), clean(m.group(2)), clean(m.group(3))

    return ("","","")

# --------- tag sets ---------
DF_TAGS      = {"didataflow","dataflow","dflow"}
JOB_TAGS     = {"dijob","dibatchjob","job","batch_job"}
PROJECT_TAGS = {"diproject","project"}
SCHEMA_TAG   = "dischema"
ELEMENT_TAG  = "dielement"
SCRIPT_FUNC  = {"discriptfunction","difunction","scriptfunction"}

# --------- parent map / context ---------
def build_parent_map(root):
    return {c: p for p in root.iter() for c in p}

def first_ancestor(node, pm, tag_set):
    cur=node
    for _ in range(300):
        cur = pm.get(cur)
        if not cur: return None
        if lower(strip_ns(cur.tag)) in tag_set: return cur
    return None

def job_name_from_node(job_node):
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch,"tag","")))=="diattribute" and lower((ch.attrib.get("name") or ""))=="job_name":
            v=(ch.attrib.get("value") or "").strip()
            if v: return v
    return (job_node.attrib.get("name") or job_node.attrib.get("displayName") or "").strip()

def map_df_to_project(root, pm):
    out={}
    for df in root.iter():
        if lower(strip_ns(getattr(df,"tag",""))) in DF_TAGS:
            dn=(df.attrib.get("name") or df.attrib.get("displayName") or "").strip()
            proj = first_ancestor(df, pm, PROJECT_TAGS)
            if dn and proj:
                out[dn] = (proj.attrib.get("name") or proj.attrib.get("displayName") or "").strip()
    # single-project fallback
    if not out:
        projs=[n for n in root.iter() if lower(strip_ns(getattr(n,"tag",""))) in PROJECT_TAGS]
        if len(projs)==1:
            pname=(projs[0].attrib.get("name") or projs[0].attrib.get("displayName") or "").strip()
            for df in root.iter():
                if lower(strip_ns(getattr(df,"tag",""))) in DF_TAGS:
                    dn=(df.attrib.get("name") or df.attrib.get("displayName") or "").strip()
                    if dn: out.setdefault(dn, pname)
    return out

def map_df_to_job(root, pm):
    out={}
    for df in root.iter():
        if lower(strip_ns(getattr(df,"tag",""))) in DF_TAGS:
            dn=(df.attrib.get("name") or df.attrib.get("displayName") or "").strip()
            job = first_ancestor(df, pm, JOB_TAGS)
            if dn and job:
                out[dn] = job_name_from_node(job)
    # single-job fallback
    if not out:
        jobs=[n for n in root.iter() if lower(strip_ns(getattr(n,"tag",""))) in JOB_TAGS]
        if len(jobs)==1:
            jn=job_name_from_node(jobs[0])
            for df in root.iter():
                if lower(strip_ns(getattr(df,"tag",""))) in DF_TAGS:
                    dn=(df.attrib.get("name") or df.attrib.get("displayName") or "").strip()
                    if dn: out.setdefault(dn, jn)
    return out

def nearest_schema(elem, pm):
    cur=elem
    for _ in range(200):
        if lower(strip_ns(getattr(cur,"tag",""))) == SCHEMA_TAG:
            return clean(cur.attrib.get("name") or cur.attrib.get("displayName") or "")
        cur = pm.get(cur)
        if not cur: break
    return "Join"

def nearest_element(elem, pm):
    cur=elem
    for _ in range(120):
        if lower(strip_ns(getattr(cur,"tag",""))) == ELEMENT_TAG:
            nm=clean(cur.attrib.get("name") or cur.attrib.get("displayName") or "")
            if nm: return nm
        cur = pm.get(cur)
        if not cur: break
    return ""

# --------- index custom functions ---------
def index_custom_functions(root):
    idx={}
    for fn in root.iter():
        if lower(strip_ns(getattr(fn,"tag",""))) not in SCRIPT_FUNC: continue
        name=(fn.attrib.get("name") or fn.attrib.get("displayName") or "").strip()
        if not name: continue
        upper=name.upper()
        found=set()
        # function_call nodes
        for call in fn.iter():
            if lower(strip_ns(getattr(call,"tag","")))=="function_call":
                an=lower(call.attrib.get("name",""))
                if an in ("lookup","lookup_ext"):
                    blob=" ".join([f'{k}="{v}"' for k,v in call.attrib.items()])+" "+gather_text(call)
                    if an=="lookup_ext":
                        ds,sch,tbl = extract_triple(blob, True)
                    else:
                        ds,sch,tbl = extract_triple(blob, False)
                    if ds and tbl: found.add( (an=="lookup_ext", clean(ds), clean(sch), clean(tbl)) )
        # raw text fallback
        txt = gather_text(fn)
        if HAS_LOOKUP.search(txt):
            ds,sch,tbl = extract_triple(txt, False)
            if ds and tbl: found.add( (False, clean(ds), clean(sch), clean(tbl)) )
        if HAS_LOOKUP_EXT.search(txt):
            ds,sch,tbl = extract_triple(txt, True)
            if ds and tbl: found.add( (True, clean(ds), clean(sch), clean(tbl)) )
        if found: idx[upper]=found
    return idx

# --------- main parse ---------
Row = namedtuple("Row", [
    "project_name","job_name","dataflow_name",
    "role","datastore","schema","table",
    "lookup_position","in_transf_used_count"
])

def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    pm   = build_parent_map(root)

    df2proj = map_df_to_project(root, pm)
    df2job  = map_df_to_job(root, pm)
    fn_idx  = index_custom_functions(root)

    rows=[]

    # per-DF buckets
    df_sources_targets = defaultdict(set)            # (df) -> {(role, ds, sch, tbl)}
    df_lookup_pos      = defaultdict(lambda: defaultdict(list))  # (df) -> {(role,ds,sch,tbl)->[pos,...]}
    df_lookup_ext      = defaultdict(lambda: defaultdict(set))   # (df) -> {(role,ds,sch,tbl)->{schema,...}}

    # walk dataflows
    for df in root.iter():
        if lower(strip_ns(getattr(df,"tag",""))) not in DF_TAGS: continue
        df_name = (df.attrib.get("name") or df.attrib.get("displayName") or "").strip()

        # capture inside this DF subtree
        for e in df.iter():
            if not isinstance(e.tag,str): continue
            tag = lower(strip_ns(e.tag))

            # sources/targets
            if tag in {"didatabasetablesource","didatabasetabletarget"}:
                a = {k.lower():v for k,v in e.attrib.items()}
                role = "source" if "source" in tag else "target"
                ds  = clean(a.get("datastorename") or a.get("datastore") or "")
                sch = clean(a.get("ownername") or a.get("schema") or a.get("owner") or "")
                tbl = clean(a.get("tablename") or a.get("table") or "")
                if ds and tbl:
                    df_sources_targets[df_name].add((role, ds, sch, tbl))
                continue

            # direct FUNCTION_CALL lookup / lookup_ext
            if tag == "function_call":
                fname = (e.attrib.get("name") or "").strip()
                lname = fname.lower()
                blob  = " ".join([f'{k}="{v}"' for k,v in e.attrib.items()])+" "+gather_text(e)
                schema_out = nearest_schema(e, pm) or "Join"

                if lname in ("lookup","lookup_ext"):
                    ds,sch,tbl = extract_triple(blob, lname=="lookup_ext")
                    if ds and tbl:
                        key=("lookup_ext" if lname=="lookup_ext" else "lookup", ds, sch, tbl)
                        if lname=="lookup_ext":
                            df_lookup_ext[df_name][key].add(schema_out)
                        else:
                            col = nearest_element(e, pm)
                            if col:
                                df_lookup_pos[df_name][key].append(f"{schema_out}>>{col}")
                else:
                    # custom DIScriptFunction call used inside DF
                    u = fname.upper()
                    if u in fn_idx:
                        col = nearest_element(e, pm)
                        schema_out = schema_out or "Join"
                        for is_ext, ds, sch, tbl in fn_idx[u]:
                            key=("lookup_ext" if is_ext else "lookup", ds, sch, tbl)
                            if is_ext:
                                df_lookup_ext[df_name][key].add(schema_out)
                            else:
                                if col:
                                    df_lookup_pos[df_name][key].append(f"{schema_out}>>{col}")
                continue

            # DIAttribute mapping text inside DF (lookup/lookup_ext or function names)
            if tag == "diattribute" and lower(e.attrib.get("name","")) in {
                "ui_mapping_text","ui_acta_from_schema_0","ui_acta_from_schema_1","ui_display_name"
            }:
                txt = e.attrib.get("value") or e.text or ""
                if HAS_LOOKUP.search(txt):
                    ds,sch,tbl = extract_triple(txt, False)
                    if ds and tbl:
                        key=("lookup", ds, sch, tbl)
                        col = nearest_element(e, pm)
                        schema_out = nearest_schema(e, pm) or "Join"
                        if col:
                            df_lookup_pos[df_name][key].append(f"{schema_out}>>{col}")
                if HAS_LOOKUP_EXT.search(txt):
                    ds,sch,tbl = extract_triple(txt, True)
                    if ds and tbl:
                        key=("lookup_ext", ds, sch, tbl)
                        schema_out = nearest_schema(e, pm) or "Join"
                        df_lookup_ext[df_name][key].add(schema_out)
                # custom function name mentions
                for fn_upper in fn_idx.keys():
                    if re.search(rf'\b{re.escape(fn_upper)}\s*\(', txt, re.I):
                        col = nearest_element(e, pm)
                        schema_out = nearest_schema(e, pm) or "Join"
                        for is_ext, ds, sch, tbl in fn_idx[fn_upper]:
                            key=("lookup_ext" if is_ext else "lookup", ds, sch, tbl)
                            if is_ext:
                                df_lookup_ext[df_name][key].add(schema_out)
                            else:
                                if col:
                                    df_lookup_pos[df_name][key].append(f"{schema_out}>>{col}")
                continue

    # materialize rows
    Row = namedtuple("Row", [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table","lookup_position","in_transf_used_count"
    ])
    rows=[]

    def proj_for(dfname): return df2proj.get(dfname, "")
    def job_for(dfname):  return df2job.get(dfname, "")

    for dfname, st in df_sources_targets.items():
        for role, ds, sch, tbl in sorted(st):
            rows.append(Row(proj_for(dfname), job_for(dfname), dfname, role,
                            clean(ds), clean(sch), clean(tbl), "", 0))

    for dfname, m in df_lookup_pos.items():
        for (role, ds, sch, tbl), pos in m.items():
            uniq = dedupe_keep_order([clean(x) for x in pos])
            rows.append(Row(proj_for(dfname), job_for(dfname), dfname, "lookup",
                            clean(ds), clean(sch), clean(tbl),
                            ", ".join(uniq), len(uniq)))

    for dfname, m in df_lookup_ext.items():
        for (role, ds, sch, tbl), schset in m.items():
            uniq = dedupe_keep_order([clean(x) for x in schset])
            rows.append(Row(proj_for(dfname), job_for(dfname), dfname, "lookup_ext",
                            clean(ds), clean(sch), clean(tbl),
                            ", ".join(uniq), len(uniq)))

    df = pd.DataFrame(rows, columns=Row._fields)

    # final cleanup / dedupe / sort
    for c in ["project_name","job_name","dataflow_name","role","datastore","schema","table","lookup_position"]:
        df[c]=df[c].map(clean)
    df = df.drop_duplicates(subset=["project_name","job_name","dataflow_name","role","datastore","schema","table","lookup_position"])
    df = df.sort_values(["project_name","job_name","dataflow_name","role","datastore","schema","table"]).reset_index(drop=True)
    return df

def main():
    out = Path(OUT_CSV)
    df = parse_xml(XML_PATH)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {out.resolve()}  |  rows: {len(df)}")

if __name__ == "__main__":
    main()
