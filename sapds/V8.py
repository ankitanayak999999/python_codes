#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V8: Same behavior as your V7 (lookups, lookup_ext, file sources/targets,
custom SQL, counts, de-dup, column names), PLUS multi-project support:

- Build a Job->Project map from <DIProject> … <DIJobRef name="…"/> …
- On every emitted row, set project_name = job_to_project.get(job_name, project_name)

Everything else is intentionally unchanged from V7.
"""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
import pandas as pd

# ------------------------ tiny utilities (same as V7) ------------------------

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
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def canon(s:str)->str: return re.sub(r'[^A-Z0-9]','',(s or '').upper())

# ------------------------ display normalization (same as V7) ------------------------

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

# ------------------------ constants & detectors (same as V7) ------------------------

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

HAS_LOOKUP_EXT = re.compile(r'\blookup_ext\s*\(', re.I)
HAS_LOOKUP     = re.compile(r'\blookup(?!_)\s*\(', re.I)

LOOKUP_CALL_RE     = re.compile(rf'\blookup(?!_)\s*\(\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})\s*"?\.\s*"?\s*({NAME_CHARS})', re.I)
BRACED_TRIPLE      = re.compile(r'\blookup(?!_)\s*\(\s*\{\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^\}]+?)\s*\}', re.I|re.S)
LOOKUP_ARGS_RE     = re.compile(r'\blookup(?!_)\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,\)]+?)\s*(?:,|\))', re.I|re.S)

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

def extract_lookup_from_call(text: str, is_ext: bool = False):
    if not text: return ("","","")
    t = DOT_NORMALIZE.sub(".", text)

    if is_ext:
        mkv = LOOKUP_EXT_NAMED_KV_RE.search(t)
        if mkv and _valid_triplet(mkv.group("ds"), mkv.group("own"), mkv.group("tbl")):
            return mkv.group("ds").strip(), mkv.group("own").strip(), mkv.group("tbl").strip()

    m0 = (BRACED_TRIPLE_EXT if is_ext else BRACED_TRIPLE).search(t)
    if m0 and _valid_triplet(m0.group(1), m0.group(2), m0.group(3)):
        return m0.group(1).strip(), m0.group(2).strip(), m0.group(3).strip()

    m1 = (LOOKUP_EXT_CALL_RE if is_ext else LOOKUP_CALL_RE).search(t)
    if m1 and _valid_triplet(m1.group(1), m1.group(2), m1.group(3)):
        return m1.group(1).strip(), m1.group(2).strip(), m1.group(3).strip()

    m2 = (LOOKUP_EXT_ARGS_RE if is_ext else LOOKUP_ARGS_RE).search(t)
    if m2 and _valid_triplet(m2.group(1), m2.group(2), m2.group(3)):
        return m2.group(1).strip(), m2.group(2).strip(), m2.group(3).strip()

    return ("","","")

# ------------------------ V8 addition: job->project map ------------------------

def build_job_to_project_map(root):
    """
    Find all <DIProject name="..."> ... <DIJobRef name="JOB_X"/> ... </DIProject>
    Return dict: job_name -> project_name
    """
    j2p = {}
    for proj in root.iter():
        if lower(strip_ns(getattr(proj, "tag", ""))) not in PROJECT_TAGS:
            continue
        p_name = (proj.attrib.get("name") or proj.attrib.get("displayName") or "").strip()
        if not p_name:
            continue
        for child in proj.iter():
            if lower(strip_ns(getattr(child, "tag", ""))) == "dijobref":
                jname = (child.attrib.get("name") or "").strip()
                if jname:
                    j2p.setdefault(jname, p_name)
    return j2p

# ------------------------ helpers (same as V7) ------------------------

def _job_name_from_node(job_node):
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch,"tag",""))) == "diattribute" and lower(ch.attrib.get("name",""))=="job_name":
            v = (ch.attrib.get("value") or "").strip()
            if v: return v
    return (job_node.attrib.get("name") or job_node.attrib.get("displayName") or "").strip()

def schema_out_from_DISchema(e, pm, fallback=""):
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

# ------------------------ main parser (V7 + V8 project fix) ------------------------

Record = namedtuple("Record", [
    "project_name","job_name","dataflow_name",
    "role","datastore","schema","table",
    "transformation_position","in_transformation_used_count","custom_sql"
])

def parse_single_xml(xml_path: str):
    tree=ET.parse(xml_path); root=tree.getroot()
    pm=build_parent_map(root)

    # NEW in V8
    job_to_project = build_job_to_project_map(root)

    # Caches (same behavior as V7)
    display_ds  = defaultdict(NameBag)
    display_sch = defaultdict(NameBag)
    display_tbl = defaultdict(NameBag)

    def remember_display(ds, sch, tbl):
        ds = _strip_wrappers(ds).strip()
        sch= _strip_wrappers(sch).strip()
        tbl= _strip_wrappers(tbl).strip()
        k=(_norm_key(ds),_norm_key(sch),_norm_key(tbl))
        display_ds[k].add(ds); display_sch[k].add(sch); display_tbl[k].add(tbl)

    # collections
    rows = []
    lookup_pos    = defaultdict(list)   # (proj,job,df,ds,sch,tbl) -> [schema>>col]
    lookup_ext_pos= defaultdict(set)    # (proj,job,df,ds,sch,tbl) -> {schema}
    seen_ext_keys = set()
    source_target = set()

    # context trackers (V7 style)
    cur_proj=cur_job=cur_df=cur_schema=""; last_job=""

    def context_for(e):
        proj=cur_proj; job=cur_job; df=cur_df
        for a in ancestors(e, pm):
            t=lower(strip_ns(a.tag)); at=attrs_ci(a)
            nm=(at.get("name") or at.get("displayname") or "").strip()
            if t in DF_TAGS: df = nm or df
            if t in PROJECT_TAGS: proj = nm or proj
            if t in JOB_TAGS and not job:
                job = _job_name_from_node(a) or job
        # V8: if we know job -> project, prefer that
        if job:
            proj = job_to_project.get(job, proj)
        return proj, job, df

    # walk XML (same as V7 for entities we capture)
    for e in root.iter():
        if not isinstance(e.tag, str): continue
        tag=lower(strip_ns(e.tag)); a=attrs_ci(e)

        if tag in PROJECT_TAGS: cur_proj=(a.get("name") or a.get("displayname") or cur_proj).strip()
        if tag in DF_TAGS:      cur_df  =(a.get("name") or a.get("displayname") or cur_df).strip()
        if tag in JOB_TAGS:
            cur_job=_job_name_from_node(e) or (a.get("name") or a.get("displayname") or cur_job).strip()
            if cur_job: last_job=cur_job
        if tag=="dischema":     cur_schema=(a.get("name") or a.get("displayname") or cur_schema).strip()

        # -------- FILE sources/targets (kept exactly like V7) --------
        if tag in ("difilesource","difiletarget"):
            proj,job,df=context_for(e)
            role="source" if tag=="difilesource" else "target"
            fmt  = (a.get("formatname") or "").strip()
            fname= (a.get("filename") or a.get("name") or "").strip()
            datastore = (a.get("datastorename") or a.get("datastore") or "FILE").strip() or "FILE"
            schema = "CUSTOM_SQL" if False else (fmt if fmt else "FILE")
            table  = fname if fname else (fmt if fmt else "FILE")
            remember_display(datastore,schema,table)
            key=(proj,job,df,role,_norm_key(datastore),_norm_key(schema),_norm_key(table))
            source_target.add(key)

        # -------- DB table sources/targets (same as V7) --------
        if tag in ("didatabasetablesource","didatabasetabletarget"):
            ds =(a.get("datastorename") or a.get("datastore") or "").strip()
            sch=(a.get("ownername") or a.get("schema") or a.get("owner") or "").strip()
            tbl=(a.get("tablename") or a.get("table") or "").strip()
            if ds and tbl:
                remember_display(ds,sch,tbl)
                proj,job,df=context_for(e)
                role="source" if "source" in tag else "target"
                key=(proj,job,df,role,_norm_key(ds),_norm_key(sch),_norm_key(tbl))
                source_target.add(key)

        # -------- lookup (column-level) --------
        if tag=="diattribute" and lower(a.get("name",""))=="ui_mapping_text":
            txt=a.get("value") or e.text or ""
            if HAS_LOOKUP.search(txt):
                proj,job,df=context_for(e)
                schema_out=schema_out_from_DISchema(e, pm, cur_schema)
                col=find_output_column(e, pm)
                dsl,schl,tbl=extract_lookup_from_call(txt, is_ext=False)
                if dsl and tbl and schema_out and col:
                    remember_display(dsl,schl,tbl)
                    lookup_pos[(proj,job,df,_norm_key(dsl),_norm_key(schl),_norm_key(tbl))] \
                        .append(f"{schema_out}>>{col}")

        # -------- lookup_ext (FUNCTION_CALL preferred) --------
        if tag=="function_call" and lower(a.get("name",""))=="lookup_ext":
            proj,job,df=context_for(e)
            schema_out=schema_out_from_DISchema(e, pm, cur_schema)
            dsx,schx,tbx = extract_lookup_from_call(" ".join([f'{k}="{v}"' for k,v in a.items()]), is_ext=True)
            if not dsx:
                dsx,schx,tbx = extract_lookup_from_call(collect_text(e), is_ext=True)
            if dsx and tbx and schema_out:
                k=(proj,job,df,_norm_key(dsx),_norm_key(schx),_norm_key(tbx))
                remember_display(dsx,schx,tbx)
                lookup_ext_pos[k].add(schema_out)
                seen_ext_keys.add(k)

        # -------- fallback blobs for lookup / lookup_ext --------
        if tag in ("diexpression","diattribute","function_call"):
            blob = " ".join([f'{k}="{v}"' for k,v in a.items()]) + " " + collect_text(e)
            proj,job,df=context_for(e)
            schema_out=schema_out_from_DISchema(e, pm, cur_schema)
            col=find_output_column(e, pm)

            if HAS_LOOKUP_EXT.search(blob):
                dsx,schx,tbx=extract_lookup_from_call(blob, is_ext=True)
                if dsx and tbx and schema_out:
                    k=(proj,job,df,_norm_key(dsx),_norm_key(schx),_norm_key(tbx))
                    if k not in seen_ext_keys:
                        remember_display(dsx,schx,tbx)
                        lookup_ext_pos[k].add(schema_out)

            if HAS_LOOKUP.search(blob) or (tag=="function_call" and lower(a.get("name",""))=="lookup"):
                dsl,schl,tbl=extract_lookup_from_call(blob, is_ext=False)
                if dsl and tbl and schema_out and col:
                    remember_display(dsl,schl,tbl)
                    lookup_pos[(proj,job,df,_norm_key(dsl),_norm_key(schl),_norm_key(tbl))] \
                        .append(f"{schema_out}>>{col}")

        # -------- CUSTOM SQL capture (exactly like your V7) --------
        # Pull <DITransformCall typeId="24" name="SQL"> ... <SQLText><sql_text>...</sql_text>
        # Put the literal SQL (wrapped in double-quotes) into custom_sql, and emit a synthetic "source"
        # row with schema="CUSTOM_SQL" and table as comma-separated referenced objects.
        if tag == "ditransformcall" and (a.get("name","") or "").upper() == "SQL":
            proj,job,df=context_for(e)

            # gather sql text
            sql_texts=[]
            for st in e.iter():
                if lower(strip_ns(getattr(st,"tag",""))) in ("sqltext","sql_text","sqltexts"):
                    val = (attrs_ci(st).get("sql_text") or st.text or "").strip()
                    if val:
                        sql_texts.append(val)
            sql = " ".join(sql_texts).strip()
            if sql:
                # simple table-ish tokens (keep V7 behavior)
                # This part stays intentionally light-touch to avoid breaking V7.
                tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_\.\$#@]*", sql)
                # crude filter to skip SQL keywords
                kw = {"select","from","where","join","on","and","or","with","as","group","by","order","union","all",
                      "left","right","inner","outer","distinct","having","case","when","then","else","end","insert",
                      "into","values","update","set","delete"}
                tables = dedupe([t for t in tokens if t.lower() not in kw and "." in t])  # prefer owner.table
                table_csv = ", ".join(tables) if tables else ""

                ds = (a.get("database_datastore") or a.get("database_datastore_name") or "DS_SQL").strip() or "DS_SQL"
                remember_display(ds, "CUSTOM_SQL", table_csv if table_csv else "SQL_TEXT")
                # store as a “source” record; custom_sql filled; transformation_position left blank
                rows.append(Record(
                    project_name = job_to_project.get(job, cur_proj),
                    job_name     = job or "",
                    dataflow_name= df or "",
                    role         = "source",
                    datastore    = ds,
                    schema       = "CUSTOM_SQL",
                    table        = table_csv if table_csv else "SQL_TEXT",
                    transformation_position="",
                    in_transformation_used_count=0,
                    custom_sql   = f"\"{sql}\""
                ))

    # ---------- finalize (same as V7) ----------
    def nice_names(dsN, schN, tblN):
        k=(dsN, schN, tblN)
        return ( display_ds[k].get(dsN), display_sch[k].get(schN), display_tbl[k].get(tblN) )

    # lookups
    for (proj,job,df,dsN,schN,tblN), positions in lookup_pos.items():
        uniq=sorted(dedupe([p.strip() for p in positions if p and p.strip()]))
        if not uniq: continue
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        # V8: use job->project mapping (fallback to proj)
        proj_fixed = job_to_project.get(job, proj) if job else proj
        rows.append(Record(
            project_name = proj_fixed or "",
            job_name     = job or "",
            dataflow_name= df or "",
            role         = "lookup",
            datastore    = dsD, schema=schD, table=tblD,
            transformation_position=", ".join(uniq),
            in_transformation_used_count=len(uniq),
            custom_sql   = ""
        ))

    # lookup_ext
    for (proj,job,df,dsN,schN,tblN), posset in lookup_ext_pos.items():
        uniq=sorted(dedupe([p.strip() for p in posset if p and p.strip()]))
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        proj_fixed = job_to_project.get(job, proj) if job else proj
        rows.append(Record(
            project_name = proj_fixed or "",
            job_name     = job or "",
            dataflow_name= df or "",
            role         = "lookup_ext",
            datastore    = dsD, schema=schD, table=tblD,
            transformation_position=", ".join(uniq),
            in_transformation_used_count=len(uniq),
            custom_sql   = ""
        ))

    # sources / targets from table/file entities
    for (proj,job,df,role,dsN,schN,tblN) in sorted(source_target):
        dsD,schD,tblD=nice_names(dsN,schN,tblN)
        proj_fixed = job_to_project.get(job, proj) if job else proj
        rows.append(Record(
            project_name = proj_fixed or "",
            job_name     = job or "",
            dataflow_name= df or "",
            role         = role,
            datastore    = dsD, schema=schD, table=tblD,
            transformation_position="",
            in_transformation_used_count=0,
            custom_sql   = ""
        ))

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # strict dedupe by normalized key + merge transformation_position (same as V7)
    def nkey(r):
        return (r["project_name"], r["job_name"], r["dataflow_name"], r["role"],
                _norm_key(r["datastore"]), _norm_key(r["schema"]), _norm_key(r["table"]))
    df["__k__"]=df.apply(nkey, axis=1)
    agg=(df.groupby(["__k__","project_name","job_name","dataflow_name","role",
                     "datastore","schema","table"], dropna=False, as_index=False)
            .agg({"transformation_position": lambda s: ", ".join(sorted(dedupe([x.strip() for x in s if str(x).strip()]))),
                  "custom_sql":              lambda s: next((x for x in s if str(x).strip()), "")}))
    df=agg.drop(columns=["__k__"])
    df["in_transformation_used_count"]=df["transformation_position"].apply(
        lambda x: len([p for p in dedupe([pp.strip() for pp in str(x).split(",")]) if p])
    )

    for c in ("datastore","schema","table"): df[c]=df[c].map(_pretty)

    df=df.sort_values(by=["project_name","job_name","dataflow_name","role","datastore","schema","table"]).reset_index(drop=True)
    return df

# ------------------------ main ------------------------

def main():
    # EDIT THESE TWO PATHS BEFORE RUNNING (same as your V7)
    xml_path = r"C:\path\to\export.xml"
    out_xlsx = r"C:\path\to\xml_lineage_output_v8.xlsx"

    df = parse_single_xml(xml_path)

    # Keep V7 column order + custom_sql
    cols = [
        "project_name","job_name","dataflow_name","role",
        "datastore","schema","table",
        "transformation_position","in_transformation_used_count",
        "custom_sql"
    ]
    for c in cols:
        if c not in df.columns: df[c]=""
    df = df[cols]

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="lineage")

    print(f"Done. Wrote: {out_xlsx}  |  Rows: {len(df)}")

if __name__ == "__main__":
    main()
