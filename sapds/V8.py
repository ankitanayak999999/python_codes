#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
import pandas as pd

# ======================= tiny utilities =======================

def strip_ns(tag):
    return re.sub(r"^\{.*\}", "", tag) if isinstance(tag, str) else ""

def lower(s):
    return (s or "").strip().lower()

def attrs_ci(e):
    return {k.lower(): (v or "") for k, v in getattr(e, "attrib", {}).items()}

def build_parent_map(root):
    return {c: p for p in root.iter() for c in p}

def ancestors(e, pm, lim=200):
    cur = e
    for _ in range(lim):
        if cur is None:
            break
        yield cur
        cur = pm.get(cur)

def collect_text(n):
    parts = []
    if hasattr(n, "attrib"):
        parts.extend([str(v) for v in n.attrib.values() if v])
    if n.text:
        parts.append(n.text)
    for c in list(n):
        parts.append(collect_text(c))
        if c.tail:
            parts.append(c.tail)
    return " ".join([p for p in parts if p])

def dedupe(seq):
    out, seen = [], set()
    for x in seq:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def canon(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())

# ======================= display normalization =======================

def _strip_wrappers(s: str) -> str:
    if s is None:
        return ""
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
    def __init__(self):
        self.best = None

    def _score(self, s):
        if not s:
            return (-1, -1, -1)
        has_space = 1 if " " in s else 0
        clean = 1 if (s == _pretty(s)) else 0
        return (has_space, clean, len(s))

    def add(self, s):
        s = _pretty(s)
        if not s:
            return
        if self.best is None or self._score(s) > self._score(self.best):
            self.best = s

    def get(self, fallback=""):
        return self.best or _pretty(fallback)

# ======================= constants & detectors =======================

DF_TAGS = ("didataflow", "dataflow", "dflow")
JOB_TAGS = ("dijob", "dibatchjob", "job", "batch_job")
PROJECT_TAGS = ("diproject", "project")
WF_TAGS = ("diworkflow", "workflow")
CALLSTEP_TAGS = ("dicallstep", "callstep")

CACHE_WORDS = {"PRE_LOAD_CACHE", "POST_LOAD_CACHE", "CACHE", "PRE_LOAD", "POST_LOAD"}
POLICY_WORDS = {"MAX", "MIN", "MAX-NS", "MIN-NS", "MAX_NS", "MIN_NS"}
RESERVED_TOKENS = set(CACHE_WORDS) | set(POLICY_WORDS) | {"NULL", "NONE"}

def _is_cache_or_policy(token: str) -> bool:
    t = (token or "").strip().strip("'").strip('"').upper().replace(" ", "_")
    return t in RESERVED_TOKENS

def _looks_like_table(token: str) -> bool:
    t = (token or "").strip().strip("'").strip('"')
    if not t or _is_cache_or_policy(t):
        return False
    if t.isdigit():
        return False
    return bool(re.search(r"[A-Za-z]", t)) and len(t) >= 3

def _valid_triplet(ds: str, sch: str, tbl: str) -> bool:
    return bool(ds) and _looks_like_table(tbl) and not _is_cache_or_policy(sch)

NAME_CHARS = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"
DOT_NORMALIZE = re.compile(r"\s*\.\s*")

HAS_LOOKUP_EXT = re.compile(r"\blookup_ext\s*\(", re.I)
HAS_LOOKUP = re.compile(r"\blookup(?!_)\s*\(", re.I)

LOOKUP_CALL_RE = re.compile(
    rf'\blookup(?!_)\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})',
    re.I,
)
BRACED_TRIPLE = re.compile(
    r"\blookup(?!_)\s*\(\s*\{\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^\}]+?)\s*\}",
    re.I | re.S,
)
LOOKUP_ARGS_RE = re.compile(
    r"\blookup(?!_)\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*(?:,|\))",
    re.I | re.S,
)

LOOKUP_EXT_CALL_RE = re.compile(
    rf'\blookup_ext\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})',
    re.I,
)
BRACED_TRIPLE_EXT = re.compile(
    r"\blookup[_\s]*ext\s*\(\s*\{\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^\}]+?)\s*\}",
    re.I | re.S,
)
LOOKUP_EXT_ARGS_RE = re.compile(
    r"\blookup_ext\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*(?:,|\))",
    re.I | re.S,
)
LOOKUP_EXT_NAMED_KV_RE = re.compile(
    r"\blookup_ext\s*\([^)]*?"
    r"(?:tableDatastore|tabledatastore)\s*=\s*([\'\"]?)(?P<ds>[^\'\",)\s]+)\1[^)]*?"
    r"(?:tableOwner|tableowner)\s*=\s*([\'\"]?)(?P<own>[^\'\",)\s]+)\3[^)]*?"
    r"(?:tableName|tablename)\s*=\s*([\'\"]?)(?P<tbl>[^\'\",)\s]+)\5",
    re.I | re.S,
)

SQL_TABLE_RE = re.compile(r"\b([A-Z_][A-Z0-9_\$#@]*)\.([A-Z_][A-Z0-9_\$#@]*)\b", re.I)

def parse_clean_triplet(ds, sch, tbl):
    ds = (ds or "").strip().strip('"').strip("'")
    sch = (sch or "").strip().strip('"').strip("'")
    tbl = (tbl or "").strip().strip('"').strip("'")
    if _is_cache_or_policy(sch):
        sch = ""
    if not ds or not tbl:
        return ("", "", "")
    return ds, sch, tbl

def extract_lookup_from_call(text: str, is_ext: bool = False):
    """Return (datastore, owner/schema, table) or ('','',''). Prefers named KV (ext)."""
    if not text:
        return ("", "", "")
    t = DOT_NORMALIZE.sub(".", text)

    if is_ext:
        mkv = LOOKUP_EXT_NAMED_KV_RE.search(t)
        if mkv:
            ds, sch, tbl = parse_clean_triplet(
                mkv.group("ds"), mkv.group("own"), mkv.group("tbl")
            )
            if _valid_triplet(ds, sch, tbl):
                return ds, sch, tbl

    m0 = (BRACED_TRIPLE_EXT if is_ext else BRACED_TRIPLE).search(t)
    if m0:
        ds, sch, tbl = parse_clean_triplet(m0.group(1), m0.group(2), m0.group(3))
        if _valid_triplet(ds, sch, tbl):
            return ds, sch, tbl

    m1 = (LOOKUP_EXT_CALL_RE if is_ext else LOOKUP_CALL_RE).search(t)
    if m1:
        ds, sch, tbl = parse_clean_triplet(m1.group(1), m1.group(2), m1.group(3))
        if _valid_triplet(ds, sch, tbl):
            return ds, sch, tbl

    m2 = (LOOKUP_EXT_ARGS_RE if is_ext else LOOKUP_ARGS_RE).search(t)
    if m2:
        ds, sch, tbl = parse_clean_triplet(m2.group(1), m2.group(2), m2.group(3))
        if _valid_triplet(ds, sch, tbl):
            return ds, sch, tbl

    return ("", "", "")

# ======================= names (project / job / df) =======================

def _job_name_from_node(job_node):
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch, "tag", ""))) == "diattribute" and lower(
            ch.attrib.get("name", "")
        ) == "job_name":
            v = (ch.attrib.get("value") or "").strip()
            if v:
                return v
    return (
        job_node.attrib.get("name")
        or job_node.attrib.get("displayName")
        or ""
    ).strip()

def collect_df_names(root):
    out = set()
    for n in root.iter():
        if lower(strip_ns(getattr(n, "tag", ""))) in DF_TAGS:
            nm = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm:
                out.add(nm)
    return out

def build_project_job_map(root):
    """Map job name -> project name via DIProject/DIJobRef."""
    job_to_project = {}
    for p in root.iter():
        if lower(strip_ns(getattr(p, "tag", ""))) in PROJECT_TAGS:
            proj = (p.attrib.get("name") or p.attrib.get("displayName") or "").strip()
            if not proj:
                continue
            for jr in p:
                if lower(strip_ns(jr.tag)) == "dijobref":
                    jn = (jr.attrib.get("name") or jr.attrib.get("displayName") or "").strip()
                    if jn:
                        job_to_project[jn] = proj
    return job_to_project

def build_df_job_map(root):
    """Map dataflow name -> job name using call graph across jobs/workflows."""
    df_names = collect_df_names(root)
    df_canon = {canon(n): n for n in df_names}
    jobs = {}
    wfs = {}
    for n in root.iter():
        t = lower(strip_ns(getattr(n, "tag", "")))
        if t in JOB_TAGS:
            nm = _job_name_from_node(n)
            if nm:
                jobs[nm] = n
        elif t in WF_TAGS:
            nm = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm:
                wfs[nm] = n

    pm = build_parent_map(root)
    edges = defaultdict(set)

    def add_edge(src_kind, src_name, dst_kind, dst_name):
        if src_name and dst_name:
            edges[(src_kind, canon(src_name))].add((dst_kind, canon(dst_name)))

    for cs in root.iter():
        if lower(strip_ns(getattr(cs, "tag", ""))) not in CALLSTEP_TAGS:
            continue
        src_kind, src_name = None, None
        cur = cs
        for _ in range(200):
            cur = pm.get(cur)
            if not cur:
                break
            t = lower(strip_ns(cur.tag))
            if t in JOB_TAGS:
                src_kind, src_name = "job", _job_name_from_node(cur)
                break
            if t in WF_TAGS:
                src_kind, src_name = "wf", (cur.attrib.get("name") or cur.attrib.get("displayName") or "")
                break
        if not src_name:
            continue

        a = attrs_ci(cs)
        tgt_type = (a.get("calledobjecttype") or a.get("type") or "").strip().lower()
        names = []
        for k in ("calledobject", "name", "object", "target", "called_object"):
            if a.get(k):
                raw = a.get(k)
                names.append(raw)
                if any(sep in raw for sep in ["/", "\\", ".", ":"]):
                    names.append(
                        raw.split("/")[-1].split("\\")[-1].split(":")[-1].split(".")[-1]
                    )

        txt = " ".join(list(a.values()) + [collect_text(cs)])

        if tgt_type in ("workflow", "diworkflow"):
            for nm in names:
                add_edge(src_kind, src_name, "wf", nm)
        elif tgt_type in ("dataflow", "didataflow"):
            for nm in names:
                add_edge(src_kind, src_name, "df", nm)
        else:
            can = canon(txt)
            for w in wfs.keys():
                if canon(w) in can:
                    add_edge(src_kind, src_name, "wf", w)
            for d in df_names:
                if canon(d) in can:
                    add_edge(src_kind, src_name, "df", d)

    df_job = {}
    for j in jobs.keys():
        start = ("job", canon(j))
        seen = {start}
        stack = [start]
        reach = set()
        while stack:
            node = stack.pop()
            for nxt in edges.get(node, ()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                kind, nm = nxt
                if kind == "df":
                    real = df_canon.get(nm)
                    if real:
                        reach.add(real)
                else:
                    stack.append(nxt)
        for d in reach:
            df_job.setdefault(d, j)
    if len(jobs) == 1:
        only = list(jobs.keys())[0]
        for d in df_names:
            df_job.setdefault(d, only)
    return df_job

# ======================= schema/column helpers =======================

def schema_out_from_DISchema(e, pm, fallback=""):
    best = None
    join = None
    for a in ancestors(e, pm, 200):
        if lower(strip_ns(getattr(a, "tag", ""))) == "dischema":
            nm = (attrs_ci(a).get("name") or "").strip()
            if nm:
                if lower(nm) != "join":
                    best = nm
                    break
                else:
                    join = nm
    return best or join or fallback

def find_output_column(e, pm):
    if lower(strip_ns(getattr(e, "tag", ""))) == "dielement":
        nm = (attrs_ci(e).get("name") or "").strip()
        if nm:
            return nm
    cur = e
    for _ in range(200):
        if cur is None:
            break
        if lower(strip_ns(cur.tag)) == "dielement":
            nm = (attrs_ci(cur).get("name") or "").strip()
            if nm:
                return nm
        cur = pm.get(cur)
    return ""

# ======================= script function index =======================

def collect_script_function_lookups(root):
    """Index DIScriptFunction: name -> {'lookup':[(ds,own,tbl)], 'lookup_ext':[(ds,own,tbl)]} parsed from FUNCTION_CALLs."""
    idx = {}
    for fn in root.iter():
        if lower(strip_ns(getattr(fn, "tag", ""))) != "discriptfunction":
            continue
        fn_name = (
            getattr(fn, "attrib", {}).get("name")
            or getattr(fn, "attrib", {}).get("displayName")
            or ""
        ).strip()
        if not fn_name:
            continue
        rec = {"lookup": [], "lookup_ext": []}
        for fc in fn.iter():
            if lower(strip_ns(getattr(fc, "tag", ""))) != "function_call":
                continue
            a = attrs_ci(fc)
            nm = lower(a.get("name", ""))
            if nm not in ("lookup", "lookup_ext"):
                continue
            is_ext = nm == "lookup_ext"
            ds, sch, tbl = ("", "", "")
            if is_ext:
                ds, sch, tbl = extract_lookup_from_call(
                    " ".join([f'{k}="{v}"' for k, v in a.items()]), is_ext=True
                )
            if not ds:
                ds, sch, tbl = extract_lookup_from_call(collect_text(fc), is_ext=is_ext)
            if ds and tbl:
                rec[nm].append((ds.strip(), sch.strip(), tbl.strip()))
        rec["lookup"] = list({(d, o, t) for (d, o, t) in rec["lookup"]})
        rec["lookup_ext"] = list({(d, o, t) for (d, o, t) in rec["lookup_ext"]})
        idx[fn_name] = rec
    return idx

# ======================= main parser =======================

Record = namedtuple(
    "Record",
    [
        "project_name",
        "job_name",
        "dataflow_name",
        "role",
        "datastore",
        "schema",
        "table",
        "transformation_position",
        "transformation_usage_count",
        "custom_sql",
    ],
)

SQL_TABLE_RE = re.compile(r"\b([A-Z_][A-Z0-9_\$#@]*)\.([A-Z_][A-Z0-9_\$#@]*)\b", re.I)

def parse_single_xml(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pm = build_parent_map(root)

    df_job_map = build_df_job_map(root)
    job_project_map = build_project_job_map(root)

    display_ds = defaultdict(NameBag)
    display_sch = defaultdict(NameBag)
    display_tbl = defaultdict(NameBag)

    def remember_display(ds, sch, tbl):
        ds = _strip_wrappers(ds).strip()
        sch = _strip_wrappers(sch).strip()
        tbl = _strip_wrappers(tbl).strip()
        k = (_norm_key(ds), _norm_key(sch), _norm_key(tbl))
        display_ds[k].add(ds)
        display_sch[k].add(sch)
        display_tbl[k].add(tbl)

    source_target = set()
    lookup_pos = defaultdict(list)  # schema>>column
    lookup_ext_pos = defaultdict(set)  # schema only
    seen_lookup_keys = set()
    seen_lookupext_keys = set()
    custom_sql_rows = []

    script_fn_index = collect_script_function_lookups(root)

    cur_project = cur_job = cur_df = cur_schema = ""
    last_job = ""

    def context_for(e):
        proj = job = df = ""
        for a in ancestors(e, pm):
            t = lower(strip_ns(a.tag))
            at = attrs_ci(a)
            nm = (at.get("name") or at.get("displayname") or "").strip()
            if not df and t in DF_TAGS:
                df = nm or df
            if t in JOB_TAGS and not job:
                job = _job_name_from_node(a) or job
        if not job and df:
            job = df_job_map.get(df, "")
        proj = job_project_map.get(job, "") or cur_project
        df = df or cur_df
        job = job or cur_job or last_job
        return proj, job, df

    # ---- walk all nodes once ----
    for e in root.iter():
        if not isinstance(e.tag, str):
            continue
        tag = lower(strip_ns(e.tag))
        a = attrs_ci(e)

        # rolling context
        if tag in PROJECT_TAGS:
            cur_project = (a.get("name") or a.get("displayname") or cur_project).strip()
        if tag in DF_TAGS:
            cur_df = (a.get("name") or a.get("displayname") or cur_df).strip()
        if tag in JOB_TAGS:
            cur_job = (
                _job_name_from_node(e)
                or (a.get("name") or a.get("displayname") or cur_job).strip()
            )
            if cur_job:
                last_job = cur_job
            pj = job_project_map.get(cur_job, "")
            if pj:
                cur_project = pj
        if tag == "dischema":
            cur_schema = (a.get("name") or a.get("displayname") or cur_schema).strip()

        # ---- DB sources / targets ----
        if tag in ("didatabasetablesource", "didatabasetabletarget"):
            ds = (a.get("datastorename") or a.get("datastore") or "").strip()
            sch = (a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl = (a.get("tablename") or a.get("table") or "").strip()
            if ds and tbl:
                remember_display(ds, sch, tbl)
                proj, job, df = context_for(e)
                role = "source" if "source" in tag else "target"
                key = (
                    proj,
                    job,
                    df,
                    role,
                    _norm_key(ds),
                    _norm_key(sch),
                    _norm_key(tbl),
                )
                source_target.add(key)

        # ---- File sources / targets ----
        if tag in ("difilesource", "difiletarget"):
            schema = (a.get("formatname") or "").strip()
            table = os.path.basename((a.get("filename") or a.get("name") or "").strip())
            ds = (a.get("database_datastore") or a.get("datastorename") or "FILE").strip()
            if schema or table:
                remember_display(ds, schema, table)
                proj, job, df = context_for(e)
                role = "source" if "source" in tag else "target"
                key = (
                    proj,
                    job,
                    df,
                    role,
                    _norm_key(ds),
                    _norm_key(schema),
                    _norm_key(table),
                )
                source_target.add(key)

        # ---- Authoritative FUNCTION_CALL lookups ----
        if tag == "function_call":
            nm = lower(a.get("name", ""))
            if nm in ("lookup", "lookup_ext"):
                proj, job, df = context_for(e)
                schema_out = schema_out_from_DISchema(e, pm, cur_schema)
                dsx = schx = tbx = ""
                if nm == "lookup_ext":
                    dsx, schx, tbx = extract_lookup_from_call(
                        " ".join([f'{k}="{v}"' for k, v in a.items()]), is_ext=True
                    )
                if not dsx:
                    dsx, schx, tbx = extract_lookup_from_call(
                        collect_text(e), is_ext=(nm == "lookup_ext")
                    )
                if dsx and tbx and schema_out:
                    remember_display(dsx, schx, tbx)
                    key = (
                        proj,
                        job,
                        df,
                        _norm_key(dsx),
                        _norm_key(schx),
                        _norm_key(tbx),
                    )
                    if nm == "lookup_ext":
                        lookup_ext_pos[key].add(schema_out)
                        seen_lookupext_keys.add(key)
                    else:
                        col = find_output_column(e, pm)
                        if col:
                            lookup_pos[key].append(f"{schema_out}>>{col}")
                            seen_lookup_keys.add(key)

        # ---- Script function call-sites + inline fallback ----
        if tag in ("diattribute", "diexpression"):
            blob = (a.get("value") or "") + " " + collect_text(e)
            if not blob:
                continue
            proj, job, df = context_for(e)
            schema_out = schema_out_from_DISchema(e, pm, cur_schema)
            col = find_output_column(e, pm)

            # 1) attach DIScriptFunction internals when invoked
            for fn_name, packs in script_fn_index.items():
                if fn_name and fn_name in blob:
                    for (ds, sch, tb) in packs.get("lookup_ext", ()):
                        remember_display(ds, sch, tb)
                        k = (
                            proj,
                            job,
                            df,
                            _norm_key(ds),
                            _norm_key(sch),
                            _norm_key(tb),
                        )
                        lookup_ext_pos[k].add(schema_out)
                        seen_lookupext_keys.add(k)
                    for (ds, sch, tb) in packs.get("lookup", ()):
                        if schema_out and col:
                            remember_display(ds, sch, tb)
                            k = (
                                proj,
                                job,
                                df,
                                _norm_key(ds),
                                _norm_key(sch),
                                _norm_key(tb),
                            )
                            lookup_pos[k].append(f"{schema_out}>>{col}")
                            seen_lookup_keys.add(k)

            # 2) strict inline fallback (only if not already captured)
            if HAS_LOOKUP_EXT.search(blob):
                dsx, schx, tbx = extract_lookup_from_call(blob, is_ext=True)
                if dsx and tbx and schema_out:
                    k = (
                        proj,
                        job,
                        df,
                        _norm_key(dsx),
                        _norm_key(schx),
                        _norm_key(tbx),
                    )
                    if k not in seen_lookupext_keys:
                        remember_display(dsx, schx, tbx)
                        lookup_ext_pos[k].add(schema_out)
                        seen_lookupext_keys.add(k)

            if HAS_LOOKUP.search(blob):
                dsl, schl, tbl = extract_lookup_from_call(blob, is_ext=False)
                if dsl and tbl and schema_out and col:
                    k = (
                        proj,
                        job,
                        df,
                        _norm_key(dsl),
                        _norm_key(schl),
                        _norm_key(tbl),
                    )
                    if k not in seen_lookup_keys:
                        remember_display(dsl, schl, tbl)
                        lookup_pos[k].append(f"{schema_out}>>{col}")
                        seen_lookup_keys.add(k)

        # ---- Custom SQL transforms ----
        if tag == "ditransformcall":
            sql_chunks = []
            disp_name = ""
            for ch in e.iter():
                t2 = lower(strip_ns(getattr(ch, "tag", "")))
                if t2 == "diattribute" and lower(ch.attrib.get("name", "")) == "ui_display_name":
                    disp_name = ch.attrib.get("value", "").strip() or disp_name
                if t2 == "sqltext":
                    for st in ch.iter():
                        if lower(strip_ns(getattr(st, "tag", ""))) == "sql_text":
                            sql_chunks.append(collect_text(st) or st.text or "")
            if sql_chunks:
                sql_text = " ".join(sql_chunks).strip()
                pairs = SQL_TABLE_RE.findall(sql_text)
                tbls = sorted({f"{s}.{t}" for (s, t) in pairs})
                proj, job, df = context_for(e)
                schema = "CUSTOM_SQL"
                table = ", ".join(tbls) if tbls else ""
                pos = disp_name or schema_out_from_DISchema(e, pm, cur_schema) or "SQL"
                custom_sql_rows.append(
                    (
                        proj,
                        job,
                        df,
                        "source",
                        "DS_SQL",
                        schema,
                        table,
                        pos,
                        1,
                        f"\"{sql_text.replace('\"', '\"\"')}\"",
                    )
                )

    # ---- finalize rows ----
    def nice_names(dsN, schN, tblN):
        k = (dsN, schN, tblN)
        return (
            display_ds[k].get(dsN),
            display_sch[k].get(schN),
            display_tbl[k].get(tblN),
        )

    Row = namedtuple(
        "Row",
        [
            "project_name",
            "job_name",
            "dataflow_name",
            "role",
            "datastore",
            "schema",
            "table",
            "transformation_position",
            "transformation_usage_count",
            "custom_sql",
        ],
    )

    rows = []

    # lookups
    for (proj, job, df, dsN, schN, tblN), positions in lookup_pos.items():
        uniq = sorted(dedupe(positions))
        if not uniq:
            continue
        dsD, schD, tblD = nice_names(dsN, schN, tblN)
        rows.append(
            Row(
                proj or "",
                job or "",
                df or "",
                "lookup",
                dsD,
                schD,
                tblD,
                ", ".join(uniq),
                len(uniq),
                "",
            )
        )

    # lookup_ext
    for (proj, job, df, dsN, schN, tblN), posset in lookup_ext_pos.items():
        uniq = sorted(dedupe(posset))
        dsD, schD, tblD = nice_names(dsN, schN, tblN)
        rows.append(
            Row(
                proj or "",
                job or "",
                df or "",
                "lookup_ext",
                dsD,
                schD,
                tblD,
                ", ".join(uniq),
                len(uniq),
                "",
            )
        )

    # sources / targets (DB + files)
    for (proj, job, df, role, dsN, schN, tblN) in sorted(source_target):
        dsD, schD, tblD = nice_names(dsN, schN, tblN)
        rows.append(
            Row(
                proj or "",
                job or "",
                df or "",
                role,
                dsD,
                schD,
                tblD,
                "",
                0,
                "",
            )
        )

    # custom SQL
    for (proj, job, df, role, ds, sch, tbl, pos, cnt, sqltxt) in custom_sql_rows:
        rows.append(
            Row(
                proj or "",
                job or "",
                df or "",
                role,
                ds,
                sch,
                tbl,
                pos,
                cnt,
                sqltxt,
            )
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # strict dedupe by normalized key; merge positions/sql
    def nkey(r):
        return (
            r["project_name"],
            r["job_name"],
            r["dataflow_name"],
            r["role"],
            _norm_key(r["datastore"]),
            _norm_key(r["schema"]),
            _norm_key(r["table"]),
        )

    df["__k__"] = df.apply(nkey, axis=1)

    def merge_pos(series):
        vals = []
        for x in series:
            if str(x).strip():
                vals.extend([p.strip() for p in str(x).split(",") if p.strip()])
        return ", ".join(sorted(dedupe(vals)))

    def merge_sql(series):
        vals = [str(x).strip() for x in series if str(x).strip()]
        return " ".join(dedupe(vals))

    agg = (
        df.groupby(
            [
                "__k__",
                "project_name",
                "job_name",
                "dataflow_name",
                "role",
                "datastore",
                "schema",
                "table",
            ],
            dropna=False,
            as_index=False,
        ).agg({"transformation_position": merge_pos, "custom_sql": merge_sql})
    )
    df = agg.drop(columns=["__k__"])
    df["transformation_usage_count"] = df["transformation_position"].apply(
        lambda x: len([p for p in dedupe([pp.strip() for pp in str(x).split(",")]) if p])
    )

    for c in (
        "datastore",
        "schema",
        "table",
        "project_name",
        "job_name",
        "dataflow_name",
        "transformation_position",
        "custom_sql",
    ):
        df[c] = df[c].map(_pretty)

    df = df.sort_values(
        by=[
            "project_name",
            "job_name",
            "dataflow_name",
            "role",
            "datastore",
            "schema",
            "table",
        ]
    ).reset_index(drop=True)
    return df

# ======================= main =======================

def main():
    # ---- set these paths ----
    xml_path = r"C:\path\to\export.xml"                  # project/job/df export (multi-project ok)
    out_xlsx = r"C:\path\to\xml_lineage_output_v8.xlsx"  # output Excel

    df = parse_single_xml(xml_path)

    cols = [
        "project_name",
        "job_name",
        "dataflow_name",
        "role",
        "datastore",
        "schema",
        "table",
        "transformation_position",
        "transformation_usage_count",
        "custom_sql",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print(f"Done. Wrote: {out_xlsx}  |  Rows: {len(df)}")

if __name__ == "__main__":
    main()
