#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V9 — SAP DS XML lineage extractor
Base: your V8 behavior, plus:
  * lookup_ext only from FUNCTION_CALL (authoritative) + dedupe by position
  * still capture inline lookup() from ui_mapping_text
  * emit lookup/lookup_ext row even if DS/owner/table not parsed (to flag gaps)
  * custom SQL uses ui_display_name for transformation_position if present
  * exact source_line for EVERY row via ElementTree.iterparse sourceline
  * robust multi-project Job->Project and DF->Project mapping
"""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
import pandas as pd
from pathlib import Path

# ------------------------ small helpers ------------------------

def lower(s): return (s or "").strip().lower()
def strip_ns(tag): return re.sub(r"^\{.*?\}", "", tag) if isinstance(tag, str) else ""
def attrs_ci(e): return {k.lower(): (v or "") for k, v in getattr(e, "attrib", {}).items()}

def _pretty(s: str) -> str:
    s = (s or "").strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_key(s: str) -> str:
    return re.sub(r"[^A-Z0-9_]", "", (s or "").upper())

def dedupe(seq):
    out, seen = [], set()
    for x in seq:
        k = str(x)
        if k in seen: continue
        seen.add(k); out.append(x)
    return out

# ------------------------ constants / regex ------------------------

DF_TAGS       = ("didataflow","dataflow")
JOB_TAGS      = ("dijob","dibatchjob","job")
PROJECT_TAGS  = ("diproject","project")

HAS_LOOKUP     = re.compile(r'\blookup\s*\(', re.I)
HAS_LOOKUP_EXT = re.compile(r'\blookup[_ ]*ext\s*\(', re.I)

NAME_CHARS = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"
DOT_NORMALIZE = re.compile(r"\s*\.\s*")

# lookup() — inline (Datastore.Owner.Table)
LOOKUP_CALL_RE = re.compile(
    rf'\blookup\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})',
    re.I)

# lookup_ext() variants
LOOKUP_EXT_CALL_RE = re.compile(
    rf'\blookup[_ ]*ext\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})',
    re.I)

LOOKUP_EXT_NAMED_KV_RE = re.compile(
    r'\blookup[_ ]*ext\s*\([^)]*?'
    r'(?:tableDatastore|tabledatastore)\s*=\s*([\'"]?)(?P<ds>[^\'",)\s]+)\1[^)]*?'
    r'(?:tableOwner|tableowner)\s*=\s*([\'"]?)(?P<own>[^\'",)\s]+)\3[^)]*?'
    r'(?:tableName|tablename)\s*=\s*([\'"]?)(?P<tbl>[^\'",)\s]+)\5',
    re.I | re.S
)

SQL_FROM_JOIN_RE = re.compile(r"\b(?:from|join)\s+([A-Za-z0-9_\.\$#@]+)", re.I)

# ------------------------ XML loader (with line numbers) ------------------------

def load_xml_with_lines(xml_path: str):
    """
    Use iterparse to ensure each element has .sourceline.
    Returns (root, all_nodes) where all_nodes is a list in document order.
    """
    it = ET.iterparse(xml_path, events=("start", "end"))
    _, root = next(it)  # first start
    nodes = []
    for ev, el in it:
        if ev == "start":
            nodes.append(el)
    return root, nodes

# ------------------------ project / job / df mapping ------------------------

def _job_name_from_node(job_node):
    # prefer explicit 'job_name' attribute if present inside
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch,"tag",""))) == "diattribute" and lower(ch.attrib.get("name",""))=="job_name":
            v = (ch.attrib.get("value") or "").strip()
            if v: return v
    # else use node name/displayName
    return (job_node.attrib.get("name") or job_node.attrib.get("displayName") or "").strip()

def collect_df_names(root):
    out=set()
    for n in root.iter():
        if lower(strip_ns(getattr(n,"tag",""))) in DF_TAGS:
            nm = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: out.add(nm)
    return out

def build_job_to_project_map(root):
    j2p={}
    for p in root.iter():
        if lower(strip_ns(getattr(p,"tag",""))) not in PROJECT_TAGS: continue
        proj = (p.attrib.get("name") or p.attrib.get("displayName") or "").strip()
        if not proj: continue
        for ref in p.iter():
            if lower(strip_ns(getattr(ref,"tag",""))) == "dijobref":
                jn=(ref.attrib.get("name") or ref.attrib.get("displayName") or "").strip()
                if jn: j2p.setdefault(jn, proj)
    return j2p

def build_df_to_project_map(root):
    df_names = collect_df_names(root)
    df_proj={}
    projects=[]
    for p in root.iter():
        if lower(strip_ns(getattr(p,"tag",""))) in PROJECT_TAGS:
            nm=(p.attrib.get("name") or p.attrib.get("displayName") or "").strip()
            if nm: projects.append((nm,p))
    for pnm,p in projects:
        for d in p.iter():
            if lower(strip_ns(getattr(d,"tag",""))) in DF_TAGS:
                dn=(d.attrib.get("name") or d.attrib.get("displayName") or "").strip()
                if dn: df_proj.setdefault(dn, pnm)
    if len(projects)==1 and projects:
        only = projects[0][0]
        for dn in df_names: df_proj.setdefault(dn, only)
    return df_proj

# ------------------------ SQL helpers ------------------------

def extract_tables_from_sql(sql_text: str):
    if not sql_text: return []
    c = " ".join(sql_text.replace("\n"," ").replace("\r"," ").split())
    hits = SQL_FROM_JOIN_RE.findall(c)
    tables=[]
    for h in hits:
        parts=h.split(".")
        if len(parts)>=2: tables.append(f"{parts[-2]}.{parts[-1]}")
        else: tables.append(parts[-1])
    return dedupe(tables)

# ------------------------ main parser ------------------------

Row = namedtuple(
    "Row",
    [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table",
        "transformation_position","transformation_usage_count","custom_sql_text",
        "source_line",
    ],
)

def parse_single_xml(xml_path: str) -> pd.DataFrame:
    root, doc_nodes = load_xml_with_lines(xml_path)

    job2proj = build_job_to_project_map(root)
    df2proj  = build_df_to_project_map(root)

    rows = []

    # keepers to dedupe lookup_ext by (df, ds, sch, tbl, position)
    seen_lookup_ext_keys = set()

    cur_proj = cur_job = cur_df = ""

    def context_for(e):
        proj = cur_proj; job = cur_job; df = cur_df
        # walk up to improve context
        cur = e
        hops = 0
        while cur is not None and hops < 100:
            t = lower(strip_ns(cur.tag))
            a = attrs_ci(cur)
            nm = (a.get("name") or a.get("displayname") or "").strip()
            if t in DF_TAGS and not df: df = nm or df
            if t in PROJECT_TAGS and not proj: proj = nm or proj
            if t in JOB_TAGS and not job: job = _job_name_from_node(cur) or job
            cur = cur.getparent() if hasattr(cur, "getparent") else None
            hops += 1
        # map proj by job/df where possible
        if not proj:
            if job and job in job2proj:
                proj = job2proj[job]
            elif df and df in df2proj:
                proj = df2proj[df]
        return proj, job, df

    # Simple ancestor traversal for ElementTree (no .getparent)
    def ancestors_chain(e, limit=100):
        # build parent map on demand (one-time)
        if not hasattr(parse_single_xml, "_parent_map"):
            pm = {c:p for p in root.iter() for c in p}
            setattr(parse_single_xml, "_parent_map", pm)
        pm = getattr(parse_single_xml, "_parent_map")
        cur = e
        for _ in range(limit):
            yield cur
            cur = pm.get(cur)
            if cur is None: break

    # -------------- walk the document once --------------
    for e in root.iter():
        if not isinstance(e.tag, str): continue
        tag = lower(strip_ns(e.tag))
        a   = attrs_ci(e)

        # update rolling context
        if tag in PROJECT_TAGS:
            cur_proj = (a.get("name") or a.get("displayname") or cur_proj).strip()
        elif tag in JOB_TAGS:
            cur_job = _job_name_from_node(e) or (a.get("name") or a.get("displayname") or cur_job).strip()
        elif tag in DF_TAGS:
            cur_df  = (a.get("name") or a.get("displayname") or cur_df).strip()

        # ---------------- sources / targets (DB) ----------------
        if tag in ("didatabasetablesource","didatabasetabletarget"):
            proj,job,df = context_for(e)
            role = "source" if "source" in tag else "target"
            ds   = (a.get("datastorename") or a.get("datastore") or "").strip()
            sch  = (a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl  = (a.get("tablename") or a.get("table") or "").strip()
            rows.append(Row(
                proj,job,df,role,
                ds,sch,tbl,
                "",0,"",
                getattr(e, "sourceline", -1),
            ))

        # ---------------- sources / targets (FILE) ----------------
        if tag in ("difilesource","difiletarget"):
            proj,job,df = context_for(e)
            role = "source" if "source" in tag else "target"
            fmt   = (a.get("formatname") or "").strip()
            fname = (a.get("filename") or a.get("name") or "").strip()
            ds    = (a.get("datastorename") or a.get("datastore") or "FILE").strip() or "FILE"
            sch   = fmt or "FILE"
            tbl   = fname or "FILE_OBJECT"
            rows.append(Row(
                proj,job,df,role,
                ds,sch,tbl,
                "",0,"",
                getattr(e, "sourceline", -1),
            ))

        # ---------------- inline lookup() from ui_mapping_text ----------------
        if tag == "diattribute" and lower(a.get("name","")) == "ui_mapping_text":
            txt = a.get("value") or e.text or ""
            if HAS_LOOKUP.search(txt):
                m = LOOKUP_CALL_RE.search(DOT_NORMALIZE.sub(".", txt))
                ds = sch = tbl = ""
                if m: ds, sch, tbl = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

                # find output column & schema name (position)
                out_col = ""
                out_schema = ""
                for up in ancestors_chain(e, 50):
                    t2 = lower(strip_ns(getattr(up,"tag","")))
                    at2= attrs_ci(up)
                    if not out_schema and t2 == "dischema":
                        nm = (at2.get("name") or "").strip()
                        if nm: out_schema = nm
                    if not out_col and t2 == "dielement":
                        nm = (at2.get("name") or "").strip()
                        if nm: out_col = nm
                pos = f"{out_schema}>>{out_col}" if out_schema and out_col else (out_schema or out_col or "")

                proj,job,df = context_for(e)
                rows.append(Row(
                    proj,job,df,"lookup",
                    ds,sch,tbl,
                    pos, 1, "",
                    getattr(e, "sourceline", -1),
                ))

            # If it clearly mentions lookup but we failed to parse the triple, emit a gap row
            elif HAS_LOOKUP.search(a.get("value","")):
                proj,job,df = context_for(e)
                rows.append(Row(
                    proj,job,df,"lookup","","","", "",1,"", getattr(e,"sourceline",-1)
                ))

        # ---------------- lookup_ext() (AUTHORITATIVE: from FUNCTION_CALL only) ----------------
        if tag == "function_call" and lower(a.get("name","")) == "lookup_ext":
            # Try to parse via named args first
            blob = " ".join([f'{k}="{v}"' for k,v in a.items()]) + " "
            blob += (e.text or "")
            for ch in e:
                blob += " " + (ch.text or "") + " " + " ".join(ch.attrib.values())

            ds = sch = tbl = ""
            mkv = LOOKUP_EXT_NAMED_KV_RE.search(blob)
            if mkv:
                ds, sch, tbl = mkv.group("ds","own","tbl")
            else:
                m2 = LOOKUP_EXT_CALL_RE.search(DOT_NORMALIZE.sub(".", blob))
                if m2:
                    ds, sch, tbl = m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()

            # position = DISchema name; if we can also find output column, append >>col
            out_schema = ""
            out_col    = ""
            for up in ancestors_chain(e, 50):
                t2 = lower(strip_ns(getattr(up,"tag",""))); at2 = attrs_ci(up)
                if not out_schema and t2 == "dischema":
                    nm=(at2.get("name") or "").strip()
                    if nm: out_schema = nm
                if not out_col and t2 == "dielement":
                    nm=(at2.get("name") or "").strip()
                    if nm: out_col = nm
            # outside-column call => only schema is expected
            pos = f"{out_schema}>>{out_col}" if (out_schema and out_col) else (out_schema or "")

            proj,job,df = context_for(e)

            # even if ds/sch/tbl missing, emit a row to flag gap
            key = (proj,job,df, _norm_key(ds), _norm_key(sch), _norm_key(tbl), pos)
            if key not in seen_lookup_ext_keys:
                seen_lookup_ext_keys.add(key)
                rows.append(Row(
                    proj,job,df,"lookup_ext",
                    ds,sch,tbl,
                    pos, 1, "",
                    getattr(e, "sourceline", -1),
                ))

        # ---------------- Custom SQL blocks ----------------
        if tag in ("sqltext","sqltexts","diquery","ditransformcall"):
            sql_text = ""
            pos_name = ""   # transformation_position

            if tag in ("sqltext","sqltexts"):
                sql_text = (a.get("sql_text") or "").strip() or (e.text or "").strip()

            # climb to find a nicer UI display name
            for up in ancestors_chain(e, 30):
                t2 = lower(strip_ns(getattr(up,"tag",""))); at2=attrs_ci(up)
                if t2 == "diattribute" and lower(at2.get("name","")) == "ui_display_name":
                    pos_name = (at2.get("value") or "").strip()
                    if pos_name: break
                if t2 == "dischema" and not pos_name:
                    pos_name = (at2.get("name") or "").strip()

            # fallback — if the current node is DITransformCall, use its 'name'
            if not pos_name and tag == "ditransformcall":
                pos_name = (a.get("name") or "").strip()

            if not sql_text:
                # try to find nested SQLText
                for ch in e.iter():
                    if lower(strip_ns(getattr(ch,"tag",""))) in ("sqltext","sql_text"):
                        sql_text = (attrs_ci(ch).get("sql_text") or ch.text or "").strip()
                        if sql_text: break

            if sql_text or pos_name:
                # tables & datastore (best effort)
                tables = extract_tables_from_sql(sql_text)
                table_csv = ", ".join(tables) if tables else "SQL_TEXT"
                ds_for_sql = ""
                for up in ancestors_chain(e, 12):
                    for ch in getattr(up, "__iter__", lambda: [])():
                        if lower(strip_ns(getattr(ch,"tag","")))=="diattribute" and lower(ch.attrib.get("name",""))=="database_datastore":
                            ds_for_sql = (ch.attrib.get("value") or "").strip()
                            if ds_for_sql: break
                    if ds_for_sql: break
                ds_for_sql = ds_for_sql or "DS_SQL"

                proj,job,df = context_for(e)
                rows.append(Row(
                    proj,job,df,"custom_sql",
                    ds_for_sql,"CUSTOM",table_csv,
                    (pos_name or "SQL"), len(tables),
                    '"' + (sql_text.replace('"','""')) + '"',
                    getattr(e, "sourceline", -1),
                ))

    # --------- to DataFrame ----------
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # fill project_name using job->project, then df->project
    def fill_proj(row):
        if row["project_name"]: return row["project_name"]
        j = row.get("job_name",""); d = row.get("dataflow_name","")
        if j and j in job2proj: return job2proj[j]
        return df2proj.get(d, "")
    df["project_name"] = df.apply(fill_proj, axis=1)

    # merge duplicates: same role, same ds/sch/tbl + same position + same DF/job/proj
    def merge_key(r):
        return (
            r["project_name"], r["job_name"], r["dataflow_name"], r["role"],
            _norm_key(r["datastore"]), _norm_key(r["schema"]), _norm_key(r["table"]),
            _norm_key(r["transformation_position"]), _norm_key(r.get("custom_sql_text",""))
        )
    df["__k__"] = df.apply(merge_key, axis=1)

    # sum usage counts; keep earliest source_line for stability
    agg = {
        "transformation_usage_count": "sum",
        "source_line": "min",
        "custom_sql_text": "first",
        "datastore": "first",
        "schema": "first",
        "table": "first",
        "transformation_position": "first",
        "project_name":"first","job_name":"first","dataflow_name":"first","role":"first",
    }
    df = (df.groupby("__k__", dropna=False, as_index=False)
            .agg(agg)
            .drop(columns=["__k__"]))

    # prettify
    for c in ("datastore","schema","table","transformation_position","custom_sql_text"):
        df[c] = df[c].map(_pretty)

    df = df.sort_values(
        by=["project_name","job_name","dataflow_name","role","datastore","schema","table","transformation_position"]
    ).reset_index(drop=True)

    return df

# ------------------------ main ------------------------

def main():
    # keep your defaults here
    xml_path = r"C:\Users\raksahu\Downloads\python\input\export_afs.xml"
    out_xlsx = r"C:\Users\raksahu\Downloads\python\input\output_v9_afs.xlsx"

    df = parse_single_xml(xml_path)

    cols = [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table",
        "transformation_position","transformation_usage_count","custom_sql_text",
        "source_line",
    ]
    for c in cols:
        if c not in df.columns: df[c] = ""
    df = df[cols]

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print(f"Done. Wrote: {out_xlsx}  |  Rows: {len(df)}")

if __name__ == "__main__":
    main()
