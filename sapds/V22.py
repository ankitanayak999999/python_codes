#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
import pandas as pd

# ---------------------------- small utils ----------------------------

def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag) if isinstance(tag, str) else ""
def lower(s): return (s or "").strip().lower()
def attrs_ci(e): return {k.lower(): (v or "") for k, v in getattr(e, "attrib", {}).items()}
def build_parent_map(root): return {c: p for p in root.iter() for c in p}

def ancestors(e, pm, lim=200):
    cur = e
    for _ in range(lim):
        if cur is None: break
        yield cur
        cur = pm.get(cur)

def collect_text(node):
    """Flatten attributes, text, children text, and tails into one string for fuzzy matching."""
    parts = []
    if hasattr(node, "attrib"):
        for _, v in node.attrib.items():
            if v: parts.append(str(v))
    if node.text: parts.append(node.text)
    for ch in list(node):
        parts.append(collect_text(ch))
        if ch.tail: parts.append(ch.tail)
    return " ".join([p for p in parts if p])

def dedupe(seq):
    out, seen = [], set()
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def canon(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', (s or '').upper())

# ---------------------------- normalization/pretty ----------------------------

def _norm_key(s: str) -> str:
    """Canonical key for dedupe: strip quotes/brackets, collapse spaces/underscores, UPPER."""
    if s is None: s = ""
    s = str(s).strip().strip('"').strip("'")
    if s.startswith("[") and s.endswith("]"): s = s[1:-1]
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s.upper()

def _pretty_display(s: str) -> str:
    """Human-friendly display for sheet."""
    if s is None: s = ""
    s = str(s).strip().strip('"').strip("'")
    if s.startswith("[") and s.endswith("]"): s = s[1:-1]
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _fix_position(pos: str) -> str:
    if pos is None: return ""
    pos = str(pos).strip().rstrip(",").strip()
    if not pos: return ""
    parts = [p.strip() for p in pos.split(">>", 1)]
    if len(parts) == 2:
        op, col = parts[0], parts[1]
        op = op[:1].upper() + op[1:] if op else op
        return f"{op}>>{col}"
    else:
        op = parts[0]
        op = op[:1].upper() + op[1:] if op else op
        return op

# pick the nicest display variant we saw for a ds/schema/table triplet
class NameBag:
    def __init__(self): self.best = None
    def _score(self, s):
        if not s: return (-1, -1, -1)
        has_space = 1 if " " in s else 0
        clean = 1 if (s == _pretty_display(s)) else 0
        return (has_space, clean, len(s))
    def add(self, s):
        s = _pretty_display(s)
        if not s: return
        if self.best is None or self._score(s) > self._score(self.best):
            self.best = s
    def get(self, fallback=""): return self.best or _pretty_display(fallback)

# ---------------------------- tag sets ----------------------------

DF_TAGS = ("didataflow", "dataflow", "dflow")
JOB_TAGS = ("dijob", "dibatchjob", "job", "batch_job")
PROJECT_TAGS = ("diproject", "project")
WF_TAGS = ("diworkflow", "workflow")
CALLSTEP_TAGS = ("dicallstep", "callstep")

# ---------------------------- name collectors ----------------------------

def collect_df_names(root):
    out = set()
    for n in root.iter():
        if lower(strip_ns(getattr(n, "tag", ""))) in DF_TAGS:
            nm = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: out.add(nm)
    return out

def collect_workflow_names(root):
    out = set()
    for n in root.iter():
        if lower(strip_ns(getattr(n, "tag", ""))) in WF_TAGS:
            nm = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: out.add(nm)
    return out

def _normalize_candidates(val):
    if not val: return []
    raw = val.strip().strip('"').strip("'")
    parts = {raw}
    for sep in ("/", "\\", ".", ":"):
        if sep in raw: parts.add(raw.split(sep)[-1])
    return [p.strip() for p in parts if p.strip()]

def _job_name_from_node(job_node):
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch, "tag", ""))) == "diattribute" and lower(ch.attrib.get("name", "")) == "job_name":
            v = (ch.attrib.get("value") or "").strip()
            if v: return v
    return (job_node.attrib.get("name") or job_node.attrib.get("displayName") or "").strip()

# ---------------------------- project map ----------------------------

def build_df_project_map(root):
    df_names = collect_df_names(root)
    df_canon = {canon(n): n for n in df_names}
    projects = []
    for pn in root.iter():
        if lower(strip_ns(getattr(pn, "tag", ""))) in PROJECT_TAGS:
            pname = (pn.attrib.get("name") or pn.attrib.get("displayName") or "").strip()
            if pname: projects.append((pname, pn))

    proj_jobrefs = defaultdict(set)
    for pname, pn in projects:
        for jref in pn.iter():
            if lower(strip_ns(getattr(jref, "tag", ""))) in ("dijobref", "jobref"):
                jn = (jref.attrib.get("name") or jref.attrib.get("displayName") or "").strip()
                if jn: proj_jobrefs[pname].add(jn)

    df_proj = {}
    # direct df inside project subtree
    for pname, pn in projects:
        for df_node in pn.iter():
            if lower(strip_ns(getattr(df_node, "tag", ""))) in DF_TAGS:
                nm = (df_node.attrib.get("name") or df_node.attrib.get("displayName") or "").strip()
                if nm: df_proj.setdefault(nm, pname)
    # fuzzy text scan in project subtree
    for pname, pn in projects:
        text = canon(collect_text(pn))
        for key, real in df_canon.items():
            if real not in df_proj and key and key in text:
                df_proj[real] = pname
    if len(projects) == 1:
        only = projects[0][0]
        for nm in df_names: df_proj.setdefault(nm, only)
    return df_proj

# ---------------------------- job map via call graph ----------------------------

def build_df_job_map(root):
    df_names = collect_df_names(root)
    df_canon = {canon(n): n for n in df_names}

    jobs = {}
    workflows = {}
    for n in root.iter():
        t = lower(strip_ns(getattr(n, "tag", "")))
        if t in JOB_TAGS:
            nm = _job_name_from_node(n)
            if nm: jobs[nm] = n
        elif t in WF_TAGS:
            nm = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: workflows[nm] = n

    parent_map = build_parent_map(root)
    edges = defaultdict(set)

    def add_edge(src_kind, src_name, dst_kind, dst_name):
        if not (src_name and dst_name): return
        edges[(src_kind, canon(src_name))].add((dst_kind, canon(dst_name)))

    for cs in root.iter():
        if lower(strip_ns(getattr(cs, "tag", ""))) not in CALLSTEP_TAGS:
            continue

        # find source job/workflow ancestor
        src_kind, src_name = None, None
        cur = cs
        for _ in range(200):
            cur = parent_map.get(cur)
            if not cur: break
            t = lower(strip_ns(getattr(cur, "tag", "")))
            if t in JOB_TAGS:
                src_kind, src_name = "job", _job_name_from_node(cur); break
            if t in WF_TAGS:
                src_kind, src_name = "wf", (cur.attrib.get("name") or cur.attrib.get("displayName") or "").strip(); break
        if not src_name: continue

        a = attrs_ci(cs)
        vals = [v for v in a.values() if v]
        txt = collect_text(cs)
        vals.append(txt)

        tgt_type_raw = (a.get("calledobjecttype") or a.get("type") or "").strip().lower()
        name_candidates = []
        for key in ("calledobject", "name", "object", "target", "called_object"):
            if a.get(key):
                name_candidates.extend(_normalize_candidates(a.get(key)))
        all_text = " ".join(vals)

        if tgt_type_raw in ("workflow", "diworkflow"):
            for cand in name_candidates:
                add_edge(src_kind, src_name, "wf", cand)
        elif tgt_type_raw in ("dataflow", "didataflow"):
            for cand in name_candidates:
                add_edge(src_kind, src_name, "df", cand)
        else:
            can_text = canon(all_text)
            for w in workflows.keys():
                if canon(w) in can_text:
                    add_edge(src_kind, src_name, "wf", w)
            for d in df_names:
                if canon(d) in can_text:
                    add_edge(src_kind, src_name, "df", d)

    df_job = {}
    for job_name in jobs.keys():
        start = ("job", canon(job_name))
        seen = set([start])
        stack = [start]
        reachable_df = set()
        while stack:
            node = stack.pop()
            for nxt in edges.get(node, ()):
                if nxt in seen: continue
                seen.add(nxt)
                kind, nm = nxt
                if kind == "df":
                    real = df_canon.get(nm)
                    if real: reachable_df.add(real)
                else:
                    stack.append(nxt)
        for df in reachable_df:
            df_job.setdefault(df, job_name)

    if len(jobs) == 1:
        only = next(iter(jobs.keys()))
        for df in df_names:
            df_job.setdefault(df, only)

    return df_job

# ---------------------------- lookup extraction ----------------------------

DOT_NORMALIZE = re.compile(r"\s*\.\s*")
NAME_CHARS = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"

LOOKUP_CALL_RE      = re.compile(rf'lookup\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)
LOOKUP_EXT_CALL_RE  = re.compile(rf'lookup_ext\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)

LOOKUP_CALL_ARGS_RE = re.compile(rf'lookup\s*\(\s*\{{?\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})', re.I|re.S)
LOOKUP_EXT_ARGS_RE  = re.compile(rf'lookup_ext\s*\(\s*\{{?\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})\s*[\'"]?\s*,\s*[\'"]?\s*({NAME_CHARS})', re.I|re.S)

BRACED_TRIPLE_EXT = re.compile(
    r'lookup[_\s]*ext\s*\(\s*\{\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*,\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*,\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*\}',
    re.I|re.S
)
BRACED_TRIPLE     = re.compile(
    r'lookup\s*\(\s*\{\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*,\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*,\s*([A-Za-z0-9_\.\-\$#@\[\]% ]+?)\s*\}',
    re.I|re.S
)

LOOKUP_ATTR_RE = re.compile(
    r'tabledatastore\s*=\s*(?P<q1>[\'"]|&quot;)?(?P<ds>[^\'"&\s]+).*?'
    r'tableowner\s*=\s*(?P<q2>[\'"]|&quot;)?(?P<own>[^\'"&\s]+).*?'
    r'tablename\s*=\s*(?P<q3>[\'"]|&quot;)?(?P<tbl>[^\'"&\s]+)',
    re.I | re.S
)

LOOKUP_EXT_NAMED_KV_RE = re.compile(
    r'lookup_ext\s*\([^)]*?'
    r'(?:tableDatastore|tabledatastore)\s*=\s*([\'"]?)(?P<ds>[^\'",)\s]+)\1[^)]*?'
    r'(?:tableOwner|tableowner)\s*=\s*([\'"]?)(?P<own>[^\'",)\s]+)\3[^)]*?'
    r'(?:tableName|tablename)\s*=\s*([\'"]?)(?P<tbl>[^\'",)\s]+)\5',
    re.I | re.S
)

def extract_lookup_from_call(text: str, is_ext: bool = False):
    if not text: return ("","","")
    m0 = (BRACED_TRIPLE_EXT if is_ext else BRACED_TRIPLE).search(text)
    if m0: return m0.group(1).strip(), m0.group(2).strip(), m0.group(3).strip()
    tnorm = DOT_NORMALIZE.sub(".", text)
    m1 = (LOOKUP_EXT_CALL_RE if is_ext else LOOKUP_CALL_RE).search(tnorm)
    if m1: return m1.group(1).strip(), m1.group(2).strip(), m1.group(3).strip()
    m2 = (LOOKUP_EXT_ARGS_RE if is_ext else LOOKUP_CALL_ARGS_RE).search(text)
    if m2: return m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()
    if is_ext:
        m3 = LOOKUP_EXT_NAMED_KV_RE.search(text)
        if m3: return m3.group("ds").strip(), m3.group("own").strip(), m3.group("tbl").strip()
    return ("","","")

def extract_lookup_from_attrs_blob(blob):
    m = LOOKUP_ATTR_RE.search(blob or "")
    return (m.group("ds"), m.group("own"), m.group("tbl")) if m else ("","","")

# ---------------------------- UDF index (for embedded lookups) ----------------------------

UDF_TAGS = ("diuserdefinedfunction","diuserfunction","difunction","dicustomfunction","discriptfunction")
UDF_CALL_RE_TEMPLATE = r'\b({NAMES})\s*\('

def index_udf_lookups(root):
    udf_tables = {}
    names = []
    for n in root.iter():
        if lower(strip_ns(getattr(n, "tag", ""))) in UDF_TAGS:
            name = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if not name: continue
            body = DOT_NORMALIZE.sub(".", collect_text(n))
            uses = set()
            for pat in (BRACED_TRIPLE_EXT, LOOKUP_EXT_CALL_RE, LOOKUP_EXT_ARGS_RE, LOOKUP_EXT_NAMED_KV_RE,
                        BRACED_TRIPLE, LOOKUP_CALL_RE, LOOKUP_CALL_ARGS_RE):
                for m in pat.finditer(body):
                    if hasattr(m, "groupdict") and "ds" in m.groupdict():
                        ds, sch, tbl = m.group("ds"), m.group("own"), m.group("tbl")
                    else:
                        ds, sch, tbl = m.group(1), m.group(2), m.group(3)
                    if ds and tbl:
                        uses.add((_norm_key(ds), _norm_key(sch), _norm_key(tbl)))
            if uses:
                key = name.upper()
                udf_tables[key] = uses
                names.append(re.escape(key))
    udf_call_re = re.compile(UDF_CALL_RE_TEMPLATE.replace("{NAMES}", "|".join(names)), re.I) if names else None
    return udf_tables, udf_call_re

# ---------------------------- main parser ----------------------------

Record = namedtuple("Record", [
    "project_name","job_name","dataflow_name",
    "role","datastore","schema","table",
    "lookup_position","in_transf_used_count"
])

def parse_single_xml(xml_path: str):
    tree = ET.parse(xml_path); root = tree.getroot()
    pm = build_parent_map(root)
    df_job_map  = build_df_job_map(root)
    df_proj_map = build_df_project_map(root)
    udf_tables, udf_call_re = index_udf_lookups(root)

    # display-name memory
    display_ds  = defaultdict(NameBag)
    display_sch = defaultdict(NameBag)
    display_tbl = defaultdict(NameBag)

    def remember_display(ds, sch, tbl):
        k = (_norm_key(ds), _norm_key(sch), _norm_key(tbl))
        display_ds[k].add(ds); display_sch[k].add(sch); display_tbl[k].add(tbl)

    # aggregators
    source_target = set()  # (proj,job,df,role,dsN,schN,tblN)
    lookup_pos    = defaultdict(list)  # (proj,job,df,dsN,schN,tblN) -> [pos...]
    lookup_ext_pos= defaultdict(set)   # (proj,job,df,dsN,schN,tblN) -> {pos}

    cur_proj=cur_job=cur_df=cur_schema=""
    last_job=""

    def context_for(e):
        proj = job = df = ""
        for a in ancestors(e, pm):
            t = lower(strip_ns(a.tag)); at = attrs_ci(a)
            nm = (at.get("name") or at.get("displayname") or "").strip()
            if not df and t in DF_TAGS: df = nm or df
            if not proj and t in PROJECT_TAGS: proj = nm or proj
            if t in JOB_TAGS and not job: job = _job_name_from_node(a) or job
        df   = df or cur_df
        proj = proj or df_proj_map.get(df, cur_proj)
        job  = job or df_job_map.get(df, cur_job or last_job)
        return proj, job, df

    def schema_out_from_DISchema(e, fallback=""):
        for a in ancestors(e, pm):
            if lower(strip_ns(a.tag)) == "dischema":
                nm = (attrs_ci(a).get("name") or "").strip()
                if nm: return nm
        return fallback

    def find_output_column(e):
        if lower(strip_ns(e.tag)) == "dielement":
            nm = (attrs_ci(e).get("name") or "").strip()
            if nm: return nm
        cur = e
        for _ in range(200):
            if cur is None: break
            if lower(strip_ns(cur.tag)) == "dielement":
                nm = (attrs_ci(cur).get("name") or "").strip()
                if nm: return nm
            cur = pm.get(cur)
        return ""

    for e in root.iter():
        if not isinstance(e.tag, str): continue
        tag = lower(strip_ns(e.tag)); a = attrs_ci(e)

        if tag in PROJECT_TAGS: cur_proj = (a.get("name") or a.get("displayname") or cur_proj).strip()
        if tag in DF_TAGS:      cur_df   = (a.get("name") or a.get("displayname") or cur_df).strip()
        if tag in JOB_TAGS:
            cur_job = _job_name_from_node(e) or (a.get("name") or a.get("displayname") or cur_job).strip()
            if cur_job: last_job = cur_job
        if tag == "dischema":   cur_schema = (a.get("name") or a.get("displayname") or cur_schema).strip()

        # sources/targets
        if tag in ("didatabasetablesource","didatabasetabletarget"):
            ds  = (a.get("datastorename") or "").strip()
            sch = (a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl = (a.get("tablename") or "").strip()
            if ds and tbl:
                remember_display(ds, sch, tbl)
                proj, job, df = context_for(e)
                role = "source" if "source" in tag else "target"
                key  = (proj, job, df, role, _norm_key(ds), _norm_key(sch), _norm_key(tbl))
                source_target.add(key)

        # useful blobs
        blob_attrs = " ".join([f'{k}="{v}"' for k, v in a.items()])
        blob_all   = blob_attrs + " " + collect_text(e)

        # ui_mapping_text → column-level lookup
        if tag == "diattribute" and lower(a.get("name","")) == "ui_mapping_text":
            txt = a.get("value") or e.text or ""
            proj, job, df = context_for(e)
            schema_out = schema_out_from_DISchema(e, cur_schema)
            col = find_output_column(e)
            dsl, schl, tbl = extract_lookup_from_call(txt, is_ext=False)
            if dsl and tbl and schema_out and col:
                remember_display(dsl, schl, tbl)
                lookup_pos[(proj,job,df,_norm_key(dsl),_norm_key(schl),_norm_key(tbl))].append(f"{schema_out}>>{col}")

        # function/expression paths
        if tag in ("diexpression","diattribute","function_call"):
            proj, job, df = context_for(e)
            schema_out = schema_out_from_DISchema(e, cur_schema)
            col = find_output_column(e)

            # try attrs blob for lookup_ext kv
            ds_kv = sch_kv = tbl_kv = ""
            if tag == "function_call":
                ds_kv = (a.get("tabledatastore") or a.get("datastorename") or "").strip()
                sch_kv= (a.get("tableowner") or a.get("ownername") or "").strip()
                tbl_kv= (a.get("tablename") or "").strip()

            # parse expressions/text for forms
            # ext first
            dsx, schx, tbx = extract_lookup_from_call(blob_all, is_ext=True)
            if not (dsx and tbx) and tag == "function_call" and ("tabledatastore" in lower(blob_all) and "tablename" in lower(blob_all)):
                dsx, schx, tbx = extract_lookup_from_attrs_blob(blob_all)
            if not (dsx and tbx) and (ds_kv or tbl_kv):
                dsx, schx, tbx = ds_kv, sch_kv, tbl_kv
            if dsx and tbx:
                remember_display(dsx, schx, tbx)
                pos = f"{schema_out}>>{col}" if (schema_out and col) else schema_out
                if pos:
                    lookup_ext_pos[(proj,job,df,_norm_key(dsx),_norm_key(schx),_norm_key(tbx))].add(pos)

            # plain lookup (column-level)
            dsl, schl, tbl = extract_lookup_from_call(blob_all, is_ext=False)
            if dsl and tbl and schema_out and col:
                remember_display(dsl, schl, tbl)
                lookup_pos[(proj,job,df,_norm_key(dsl),_norm_key(schl),_norm_key(tbl))].append(f"{schema_out}>>{col}")

            # UDF calls: if name matches and that UDF uses lookups, attribute them (no column names)
            if udf_call_re and udf_call_re.search(blob_all):
                for udf_name, uses in udf_tables.items():
                    if re.search(rf'\b{re.escape(udf_name)}\s*\(', blob_all, re.I):
                        for dsN, schN, tblN in uses:
                            remember_display(dsN, schN, tblN)
                            pos = schema_out if schema_out else ""
                            if pos:
                                lookup_ext_pos[(proj,job,df,dsN,schN,tblN)].add(pos)

    # helper to get best display names
    def nice_names(dsN, schN, tblN):
        k = (dsN, schN, tblN)
        return (
            display_ds[k].get(dsN),
            display_sch[k].get(schN),
            display_tbl[k].get(tblN)
        )

    # emit rows (we dedupe by normalized ids; display uses nicest variant)
    out = []

    for (proj,job,df,dsN,schN,tblN), positions in lookup_pos.items():
        uniq = sorted(dedupe([_fix_position(p) for p in positions]))
        if not uniq: continue
        dsD, schD, tblD = nice_names(dsN, schN, tblN)
        out.append(Record(proj or "", job or "", df or "", "lookup",
                          dsD, schD, tblD, ", ".join(uniq), len(uniq)))

    for (proj,job,df,dsN,schN,tblN), posset in lookup_ext_pos.items():
        uniq = sorted(dedupe([_fix_position(p) for p in posset]))
        dsD, schD, tblD = nice_names(dsN, schN, tblN)
        out.append(Record(proj or "", job or "", df or "", "lookup_ext",
                          dsD, schD, tblD, ", ".join(uniq), len(uniq)))

    for (proj,job,df,role,dsN,schN,tblN) in sorted(source_target):
        dsD, schD, tblD = nice_names(dsN, schN, tblN)
        out.append(Record(proj or "", job or "", df or "", role,
                          dsD, schD, tblD, "", 0))

    # to DataFrame
    df = pd.DataFrame(out, columns=list(Record._fields))

    # final cleanup/dedupe on display triplets
    # normalized key to collapse duplicates that still slipped in
    def row_key(r):
        return (r["project_name"], r["job_name"], r["dataflow_name"], r["role"],
                _norm_key(r["datastore"]), _norm_key(r["schema"]), _norm_key(r["table"]))

    if not df.empty:
        df["__key__"] = df.apply(row_key, axis=1)

        # merge duplicate lookup_position and max count
        agg = (df
               .groupby(["__key__", "project_name","job_name","dataflow_name","role",
                         "datastore","schema","table"], dropna=False, as_index=False)
               .agg({
                    "lookup_position": lambda s: ", ".join(sorted(dedupe([_fix_position(x) for x in s if str(x).strip()]))),
                    "in_transf_used_count": "max"
               }))

        # split back to columns
        df = agg.drop(columns=["__key__"])

        # recompute count from positions to be safe
        def recalc_cnt(x):
            if not x or not str(x).strip(): return 0
            return len([p for p in dedupe([pp.strip() for pp in str(x).split(",")]) if p])

        df["in_transf_used_count"] = df["lookup_position"].map(recalc_cnt)

        # pretty display one more time
        for col in ("datastore","schema","table"):
            df[col] = df[col].map(_pretty_display)

        # sort
        df = df.sort_values(by=["project_name","job_name","dataflow_name","role","datastore","schema","table"]).reset_index(drop=True)

    return df

# ---------------------------- main ----------------------------

def main():
    # <<< EDIT THESE TWO PATHS >>>
    xml_path  = r"C:\path\to\your\project_or_job_export.xml"
    out_xlsx  = r"C:\path\to\export_xml_lineage_final.xlsx"

    df = parse_single_xml(xml_path)

    # Reorder to EXACT template
    want_cols = ["project_name","job_name","dataflow_name","role",
                 "datastore","schema","table","lookup_position","in_transf_used_count"]
    for c in want_cols:
        if c not in df.columns: df[c] = ""
    df = df[want_cols]

    # Write Excel
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print(f"Done. Rows: {len(df)}\nSaved: {out_xlsx}")

if __name__ == "__main__":
    main()
