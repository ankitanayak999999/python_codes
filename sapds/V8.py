#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
import pandas as pd

# ------------------------ tiny utilities ------------------------

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
        if cur is None: break
        yield cur
        cur = pm.get(cur)

def collect_text(n):
    parts=[]
    if hasattr(n,"attrib"): parts.extend([str(v) for v in n.attrib.values() if v])
    if n.text: parts.append(n.text)
    for c in list(n):
        parts.append(collect_text(c))
        if c.tail: parts.append(c.tail)
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
            seen.add(s); out.append(s)
    return out

def canon(s: str) -> str:
    return re.sub(r'[^A-Z0-9]+','', (s or '').upper())

# ------------------------ display normalization ------------------------

def _strip_wrappers(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().strip('"').strip("'")
    # remove wrapping [] or {}
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

# ------------------------ constants & detectors ------------------------

DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")
CALLSTEP_TAGS = ("dicallstep","callstep")

CACHE_WORDS  = {"PRE_LOAD_CACHE","POST_LOAD_CACHE","CACHE","PRE_LOAD","POST_LOAD"}
POLICY_WORDS = {"MAX","MIN","MAX-NS","MIN-NS","MAX_NS","MIN_NS"}

def _is_cache_or_policy(token: str) -> bool:
    t = (token or "").strip().strip("'").strip('"').upper().replace(" ","_")
    return t in CACHE_WORDS or t in POLICY_WORDS

def _looks_like_table(token: str) -> bool:
    t = (token or "").strip().strip("'").strip('"')
    if not t or _is_cache_or_policy(t): return False
    if t.isdigit(): return False
    return bool(re.search(r"[A-Za-z]", t)) and len(t) >= 3

def _valid_triplet(ds: str, sch: str, tbl: str) -> bool:
    return bool(ds) and _looks_like_table(tbl) and not _is_cache_or_policy(sch)

NAME_CHARS    = r"[A-Za-z0-9_\.\-\$#@\[\]% ]+"
DOT_NORMALIZE = re.compile(r"\s*\.\s*")

# classification (lookup_ext FIRST; lookup excludes lookup_ext via (?!_))
HAS_LOOKUP_EXT = re.compile(r'\blookup_ext\s*\(', re.I)
HAS_LOOKUP     = re.compile(r'\blookup(?!_)\s*\(', re.I)

# lookup dotted/braced/args
LOOKUP_CALL_RE     = re.compile(rf'\blookup(?!_)\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)
BRACED_TRIPLE      = re.compile(r'\blookup(?!_)\s*\(\s*\{\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^\}]+?)\s*\}', re.I|re.S)
LOOKUP_ARGS_RE     = re.compile(r'\blookup(?!_)\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*(?:,|\))', re.I|re.S)

# lookup_ext dotted/braced/args + FUNCTION_CALL named KV (most accurate)
LOOKUP_EXT_CALL_RE = re.compile(rf'\blookup_ext\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)
BRACED_TRIPLE_EXT  = re.compile(r'\blookup[_\s]*ext\s*\(\s*\{\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^\}]+?)\s*\}', re.I|re.S)
LOOKUP_EXT_ARGS_RE = re.compile(r'\blookup_ext\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*(?:,|\))', re.I|re.S)
LOOKUP_EXT_NAMED_KV_RE = re.compile(
    r'\blookup_ext\s*\([^)]*?'
    r'(?:tableDatastore|tabledatastore)\s*=\s*([\'"]?)(?P<ds>[^\'",)\s]+)\1[^)]*?'
    r'(?:tableOwner|tableowner)\s*=\s*([\'"]?)(?P<own>[^\'",)\s]+)\3[^)]*?'
    r'(?:tableName|tablename)\s*=\s*([\'"]?)(?P<tbl>[^\'",)\s]+)\5',
    re.I | re.S
)

SQL_TABLE_RE = re.compile(r'\b([A-Z_][A-Z0-9_\$#@]*)\.([A-Z_][A-Z0-9_\$#@]*)\b', re.I)

def extract_lookup_from_call(text: str, is_ext: bool = False):
    """Return (datastore, owner/schema, table) or ('','','')."""
    if not text: return ("","","")
    t = DOT_NORMALIZE.sub(".", text)

    # 1) FUNCTION_CALL named kv (ext only)
    if is_ext:
        mkv = LOOKUP_EXT_NAMED_KV_RE.search(t)
        if mkv and _valid_triplet(mkv.group("ds"), mkv.group("own"), mkv.group("tbl")):
            return mkv.group("ds").strip(), mkv.group("own").strip(), mkv.group("tbl").strip()

    # 2) braced {...}
    m0 = (BRACED_TRIPLE_EXT if is_ext else BRACED_TRIPLE).search(t)
    if m0 and _valid_triplet(m0.group(1), m0.group(2), m0.group(3)):
        return m0.group(1).strip(), m0.group(2).strip(), m0.group(3).strip()

    # 3) dotted ds.owner.table
    m1 = (LOOKUP_EXT_CALL_RE if is_ext else LOOKUP_CALL_RE).search(t)
    if m1 and _valid_triplet(m1.group(1), m1.group(2), m1.group(3)):
        return m1.group(1).strip(), m1.group(2).strip(), m1.group(3).strip()

    # 4) simple args triple
    m2 = (LOOKUP_EXT_ARGS_RE if is_ext else LOOKUP_ARGS_RE).search(t)
    if m2 and _valid_triplet(m2.group(1), m2.group(2), m2.group(3)):
        return m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()

    return ("","","")

# ------------------------ names (project / job / df) ------------------------

def _job_name_from_node(job_node):
    # prefer DIAttribute name="job_name"
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch,"tag",""))) == "diattribute" and lower(ch.attrib.get("name",""))=="job_name":
            v = (ch.attrib.get("value") or "").strip()
            if v: return v
    # fallback to node attribute
    return (job_node.attrib.get("name") or job_node.attrib.get("displayName") or "").strip()

def collect_df_names(root):
    out=set()
    for n in root.iter():
        if lower(strip_ns(getattr(n,"tag",""))) in DF_TAGS:
            nm = (n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: out.add(nm)
    return out

def build_project_job_map(root):
    """Map job name -> project name via DIProject/DIJobRef."""
    job_to_project={}
    for p in root.iter():
        if lower(strip_ns(getattr(p,"tag",""))) in PROJECT_TAGS:
            proj = (p.attrib.get("name") or p.attrib.get("displayName") or "").strip()
            if not proj: continue
            for jr in p:
                if lower(strip_ns(jr.tag))=="dijobref":
                    jn = (jr.attrib.get("name") or jr.attrib.get("displayName") or "").strip()
                    if jn: job_to_project[jn]=proj
    return job_to_project

def build_df_job_map(root):
    """Map dataflow name -> job name (via containment and call edges)."""
    df_names = collect_df_names(root); df_canon={canon(n):n for n in df_names}
    jobs={}; wfs={}
    for n in root.iter():
        t=lower(strip_ns(getattr(n,"tag","")))
        if t in JOB_TAGS:
            nm=_job_name_from_node(n)
            if nm: jobs[nm]=n
        elif t in WF_TAGS:
            nm=(n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: wfs[nm]=n

    pm=build_parent_map(root)
    edges=defaultdict(set)

    def add_edge(src_kind, src_name, dst_kind, dst_name):
        if src_name and dst_name: edges[(src_kind, canon(src_name))].add((dst_kind, canon(dst_name)))

    for cs in root.iter():
        if lower(strip_ns(getattr(cs,"tag",""))) not in CALLSTEP_TAGS: continue

        # find source (job/workflow) up the tree
        src_kind, src_name = None, None
        cur=cs
        for _ in range(200):
            cur=pm.get(cur); 
            if not cur: break
            t=lower(strip_ns(cur.tag))
            if t in JOB_TAGS: src_kind,src_name="job",_job_name_from_node(cur); break
            if t in WF_TAGS:  src_kind,src_name="wf",(cur.attrib.get("name") or cur.attrib.get("displayName") or ""); break
        if not src_name: continue

        a=attrs_ci(cs); vals=list(a.values()); vals.append(collect_text(cs))
        tgt_type=(a.get("calledobjecttype") or a.get("type") or "").strip().lower()
        names=[]
        for k in ("calledobject","name","object","target","called_object"):
            if a.get(k):
                raw=a.get(k); names.append(raw)
                if any(sep in raw for sep in ["/","\\",".",":"]):
                    names.append(raw.split("/")[-1].split("\\")[-1].split(":")[-1].split(".")[-1])

        txt=" ".join(vals)

        if tgt_type in ("workflow","diworkflow"):
            for nm in names: add_edge(src_kind,src_name,"wf",nm)
        elif tgt_type in ("dataflow","didataflow"):
            for nm in names: add_edge(src_kind,src_name,"df",nm)
        else:
            can=canon(txt)
            for w in wfs.keys():
                if canon(w) in can: add_edge(src_kind,src_name,"wf",w)
            for d in df_names:
                if canon(d) in can: add_edge(src_kind,src_name,"df",d)

    df_job={}
    for j in jobs.keys():
        start=("job", canon(j)); seen={start}; stack=[start]; reach=set()
        while stack:
            node=stack.pop()
            for nxt in edges.get(node,()):
                if nxt in seen: continue
                seen.add(nxt)
                kind, nm = nxt
                if kind=="df":
                    real=df_canon.get(nm)
                    if real: reach.add(real)
                else:
                    stack.append(nxt)
        for d in reach: df_job.setdefault(d, j)
    if len(jobs)==1:
        only=list(jobs.keys())[0]
        for d in df_names: df_job.setdefault(d, only)
    return df_job

# ------------------------ schema/column helpers ------------------------

def schema_out_from_DISchema(e, pm, fallback=""):
    """Nearest DISchema; prefer non-'Join'."""
    best=None; join=None
    for a in ancestors(e, pm, 200):
        if lower(strip_ns(getattr(a,"tag","")))=="dischema":
            nm=(attrs_ci(a).get("name") or "").strip()
            if nm:
                if lower(nm)!="join": best=nm; break
                else: join=nm
    return best or join or fallback

def find_output_column(e, pm):
    if lower(strip_ns(getattr(e,"tag","")))=="dielement":
        nm=(attrs_ci(e).get("name") or "").strip()
        if nm: return nm
    cur=e
    for _ in range(200):
        if cur is None: break
        if lower(strip_ns(cur.tag))=="dielement":
            nm=(attrs_ci(cur).get("name") or "").strip()
            if nm: return nm
        cur=pm.get(cur)
    return ""

# ------------------------ script function index ------------------------

def collect_script_function_lookups(root):
    """Index DIScriptFunction: name -> {'lookup':[(ds,own,tbl)], 'lookup_ext':[(ds,own,tbl)]} parsed from FUNCTION_CALLs."""
    idx = {}
    for fn in root.iter():
        if lower(strip_ns(getattr(fn,"tag",""))) != "discriptfunction": 
            continue
        fn_name = (getattr(fn,'attrib',{}).get("name") or getattr(fn,'attrib',{}).get("displayName") or "").strip()
        if not fn_name: 
            continue
        rec = {"lookup":[], "lookup_ext":[]}
        for fc in fn.iter():
            if lower(strip_ns(getattr(fc,"tag",""))) != "function_call":
                continue
            a = attrs_ci(fc)
            nm = lower(a.get("name",""))
            if nm not in ("lookup","lookup_ext"): 
                continue
            is_ext = (nm=="lookup_ext")
            # prefer attributes (named kv) then inner text
            ds,sch,tbl = ("","","")
            if is_ext:
                ds,sch,tbl = extract_lookup_from_call(" ".join([f'{k}="{v}"' for k,v in a.items()]), is_ext=True)
            if not ds:
                ds,sch,tbl = extract_lookup_from_call(collect_text(fc), is_ext=is_ext)
            if ds and tbl:
                rec[nm].append((ds.strip(), sch.strip(), tbl.strip()))
        # dedupe inside function
        rec["lookup"] = list({(d,o,t) for (d,o,t) in rec["lookup"]})
        rec["lookup_ext"] = list({(d,o,t) for (d,o,t) in rec["lookup_ext"]})
        idx[fn_name] = rec
    return idx

# ------------------------ main parser ------------------------

Record = namedtuple("Record", [
    "project_name","job_name","dataflow_name",
    "role","datastore","schema","table",
    "transformation_position","transformation_usage_count","custom_sql"
])

def parse_single_xml(xml_path: str):
    tree=ET.parse(xml_path); root=tree.getroot()
    pm=build_parent_map(root)

    # mappings
    df_job_map     = build_df_job_map(root)
    job_project_map= build_project_job_map(root)

    # pretty name caches
    display_ds  = defaultdict(NameBag)
    display_sch = defaultdict(NameBag)
    display_tbl = defaultdict(NameBag)

    def remember_display(ds, sch, tbl):
        ds = _strip_wrappers(ds).strip()
        sch= _strip_wrappers(sch).strip()
        tbl= _strip_wrappers(tbl).strip()
        k=(_norm_key(ds),_norm_key(sch),_norm_key(tbl))
        display_ds[k].add(ds); display_sch[k].add(sch); display_tbl[k].add(tbl)

    source_target=set()
    lookup_pos    = defaultdict(list)  # schema>>column strings
    lookup_ext_pos= defaultdict(set)   # schema strings (no column)
    seen_ext_keys = set()              # to avoid dup when both FUNCTION_CALL and DIExpression exist

    custom_sql_rows = []               # will be merged later

    # pre-index DIScriptFunction lookups
    script_fn_index = collect_script_function_lookups(root)

    cur_project=cur_job=cur_df=cur_schema=""; last_job=""

    def context_for(e):
        proj=job=df=""
        for a in ancestors(e, pm):
            t=lower(strip_ns(a.tag)); at=attrs_ci(a)
            nm=(at.get("name") or at.get("displayname") or "").strip()
            if not df and t in DF_TAGS: df=nm or df
            if t in JOB_TAGS and not job: job=_job_name_from_node(a) or job
        if not job and df: job = df_job_map.get(df,"")
        proj = job_project_map.get(job,"") or cur_project
        df = df or cur_df
        job = job or cur_job or last_job
        return proj, job, df

    # --- scan all nodes once ---
    for e in root.iter():
        if not isinstance(e.tag, str): continue
        tag=lower(strip_ns(e.tag)); a=attrs_ci(e)

        # rolling context
        if tag in PROJECT_TAGS:
