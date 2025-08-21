# sapds_column_mappings_v3_plus_st_helpers_readable_rows.py
# v3 + helpers for global DITables + Source/Target collection
# Strict 3-key link (DATASTORE, OWNER, TABLE) for database sources/targets
# Readable emits: build dict on one line, then emit(rows, **row)

import re, os, glob, html, datetime
import pandas as pd
from lxml import etree as ET
from collections import defaultdict

# -------------------- small utils --------------------
def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag) if isinstance(tag, str) else ""
def lower(s): return (s or "").strip().lower()
def attrs_ci(e): return {k.lower(): (v or "") for k, v in (getattr(e, "attrib", {}) or {}).items()}
def build_parent_map(root): return {c: p for p in root.iter() for c in p}
def ancestors(e, pm, lim=200):
    cur = e
    for _ in range(lim):
        if cur is None: break
        yield cur
        cur = pm.get(cur)

def _canon(s: str) -> str:
    """Uppercase, trim quotes/spaces, collapse whitespace."""
    s = (s or "").strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s.upper()

def emit(rows, **kwargs):
    """Build + append a row (keeps callsites clean)."""
    row = dict(kwargs)  # build
    rows.append(row)    # append

# ---------- tag sets ----------
DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")
CALLSTEP_TAGS = ("dicallstep","callstep")

DB_SOURCE_TAGS   = ("didatabasetablesource","ditablesource")
DB_TARGET_TAGS   = ("didatabasetabletarget","ditabletarget")
FILE_SOURCE_TAGS = ("difilesource",)
FILE_TARGET_TAGS = ("difiletarget",)
XLS_SOURCE_TAGS  = ("diexcelsource",)
XLS_TARGET_TAGS  = ("diexceltarget",)

ALL_SOURCE_TAGS = DB_SOURCE_TAGS + FILE_SOURCE_TAGS + XLS_SOURCE_TAGS
ALL_TARGET_TAGS = DB_TARGET_TAGS + FILE_TARGET_TAGS + XLS_TARGET_TAGS
ALL_ST_TAGS     = ALL_SOURCE_TAGS + ALL_TARGET_TAGS

# -------------------- v3 context helpers (unchanged logic) --------------------
def _job_name_from_node(job_node):
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch,"tag",""))) == "diattribute" and lower(ch.attrib.get("name",""))=="job_name":
            v = (ch.attrib.get("value") or "").strip()
            if v: return v
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
        if lower(strip_ns(getattr(p,"tag",""))) not in PROJECT_TAGS:
            continue
        proj=(p.attrib.get("name") or p.attrib.get("displayName") or "").strip()
        if not proj: continue
        for ref in p.iter():
            if lower(strip_ns(getattr(ref,"tag",""))) == "dijobref":
                jn=(ref.attrib.get("name") or ref.attrib.get("displayName") or "").strip()
                if jn: j2p.setdefault(jn, proj)
    return j2p

def build_df_project_map(root):
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
    if len(projects)==1:
        only=projects[0][0]
        for dn in df_names: df_proj.setdefault(dn, only)
    return df_proj

def build_df_job_map(root):
    pm={c:p for p in root.iter() for c in p}
    df_names = collect_df_names(root)
    df_canon = {re.sub(r'[^A-Z0-9]','',n.upper()): n for n in df_names}

    jobs = {}
    wfs  = {}
    def canon(s): return re.sub(r'[^A-Z0-9]','',(s or '').upper())

    for n in root.iter():
        t=lower(strip_ns(getattr(n,"tag","")))
        if t in JOB_TAGS:
            nm=_job_name_from_node(n)
            if nm: jobs[nm]=n
        elif t in WF_TAGS:
            nm=(n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: wfs[nm]=n

    from collections import defaultdict as dd
    edges=dd(set)
    def add_edge(src_kind, src_name, dst_kind, dst_name):
        if src_name and dst_name:
            edges[(src_kind, canon(src_name))].add((dst_kind, canon(dst_name)))

    for cs in root.iter():
        if lower(strip_ns(getattr(cs,"tag",""))) not in CALLSTEP_TAGS: continue
        cur=cs; src_kind=src_name=None
        for _ in range(200):
            cur=pm.get(cur)
            if cur is None: break
            t=lower(strip_ns(cur.tag))
            if t in JOB_TAGS: src_kind,src_name="job",_job_name_from_node(cur); break
            if t in WF_TAGS:  src_kind,src_name="wf",(cur.attrib.get("name") or cur.attrib.get("displayName") or ""); break
        if not src_name: continue

        a=attrs_ci(cs)
        tgt_type=(a.get("calledobjecttype") or a.get("type") or "").strip().lower()
        names=[]
        for k in ("calledobject","name","object","target","called_object"):
            if a.get(k):
                raw=a.get(k); names.append(raw)
                if any(sep in raw for sep in ["/","\\",".",":"]):
                    names.append(raw.split("/")[-1].split("\\")[-1].split(":")[-1].split(".")[-1])

        txt=" ".join([*a.values()])

        if tgt_type in ("workflow","diworkflow"):
            for nm in names: add_edge("job" if src_kind=="job" else "wf", src_name, "wf", nm)
        elif tgt_type in ("dataflow","didataflow"):
            for nm in names: add_edge("job" if src_kind=="job" else "wf", src_name, "df", nm)
        else:
            for w in wfs.keys():
                if canon(w) in canon(txt): add_edge("wf", src_name, "wf", w)
            for d in df_names:
                if canon(d) in canon(txt): add_edge("wf", src_name, "df", d)

    df_job={}
    for j in jobs.keys():
        start=("job", re.sub(r'[^A-Z0-9]','',j.upper()))
        seen={start}; stack=[start]; reach=set()
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

def in_dataflow(e, pm):
    for a in ancestors(e, pm, 200):
        if lower(strip_ns(getattr(a,"tag",""))) in DF_TAGS:
            return True
    return False

def nearest_schema_name(e, pm):
    for a in ancestors(e, pm, 200):
        if lower(strip_ns(getattr(a,"tag",""))) == "dischema":
            nm = (attrs_ci(a).get("name") or attrs_ci(a).get("displayname") or "").strip()
            if nm: return nm
    return ""

# ======================================================================
# 1) GLOBAL DITABLE CATALOG
# ======================================================================
def extract_ditable_catalog(root):
    """
    Scan *global* <DITable> definitions and build a strict 3-key catalog:
        (DATASTORE, OWNER, TABLE) -> set(column_names)
    """
    by3 = defaultdict(set)
    for tbl in root.iter():
        if lower(strip_ns(getattr(tbl,"tag",""))) != "ditable":
            continue
        a = attrs_ci(tbl)
        ds   = _canon(a.get("datastore")    or a.get("datastorename") or "")
        owner= _canon(a.get("owner")        or a.get("ownername")     or "")
        tnam = _canon(a.get("name")         or a.get("tablename")     or a.get("table_name") or "")
        if not (ds and owner and tnam):
            continue  # strict 3-key

        cols=set()
        for c in tbl.iter():
            if lower(strip_ns(getattr(c,"tag",""))) == "dicolumn":
                cn = (attrs_ci(c).get("name") or "").strip()
                if cn: cols.add(_canon(cn))
        if cols:
            by3[(ds, owner, tnam)] |= cols
    return by3

# ======================================================================
# 2) PER-DF SOURCE/TARGET REFERENCES
# ======================================================================
def collect_source_target_refs(root):
    """
    Walk each Dataflow → DISchema and capture Source/Target nodes (DB/File/Excel).
    Returns list of dicts with DF/Schema context + cleaned keys for DB.
    """
    pm = build_parent_map(root)
    refs=[]
    for n in root.iter():
        tag=lower(strip_ns(getattr(n,"tag","")))
        if tag not in ALL_ST_TAGS:
            continue
        a=attrs_ci(n)

        # find DF + Schema context
        df_name=schema_name=""
        for anc in ancestors(n, pm, 200):
            t=lower(strip_ns(getattr(anc,"tag","")))
            at=attrs_ci(anc)
            if not df_name and t in DF_TAGS:
                df_name=(at.get("name") or at.get("displayname") or "").strip()
            if not schema_name and t=="dischema":
                schema_name=(at.get("name") or at.get("displayname") or "").strip()

        role = "SOURCE" if tag in ALL_SOURCE_TAGS else "TARGET"
        if   tag in DB_SOURCE_TAGS + DB_TARGET_TAGS: kind="DATABASE"
        elif tag in FILE_SOURCE_TAGS + FILE_TARGET_TAGS: kind="FILE"
        elif tag in XLS_SOURCE_TAGS  + XLS_TARGET_TAGS:  kind="EXCEL"
        else: kind="UNKNOWN"

        rec = {
            "df": df_name,
            "schema": schema_name,
            "role": role,
            "kind": kind,
            # cleaned keys for DB only:
            "dsC": _canon(a.get("datastorename") or a.get("datastore") or ""),
            "ownerC": _canon(a.get("ownername") or a.get("owner") or a.get("schema") or ""),
            "tableC": _canon(a.get("tablename") or a.get("table") or a.get("name") or ""),
        }
        refs.append(rec)
    return refs

# ======================================================================
# 3) MAIN COLUMN MAPPING
# ======================================================================
def parse_single_xml(xml_path: str) -> pd.DataFrame:
    parser = ET.XMLParser(huge_tree=True, recover=True)
    root   = ET.parse(xml_path, parser=parser).getroot()
    pm     = build_parent_map(root)

    # context maps (same as your v3 approach)
    job_to_project = build_job_to_project_map(root)
    df_to_project  = build_df_project_map(root)
    df_to_job      = build_df_job_map(root)

    cur_proj = cur_job = cur_df = ""

    def context_for(e):
        proj=job=df=""
        for a in ancestors(e, pm, 200):
            t  = lower(strip_ns(getattr(a,"tag","")))
            at = attrs_ci(a)
            nm = (at.get("name") or at.get("displayname") or "").strip()
            if not df and t in DF_TAGS: df = nm or df
            if not proj and t in PROJECT_TAGS: proj = nm or proj
            if t in JOB_TAGS and not job: job = _job_name_from_node(a) or job
        df  = df or cur_df
        job = job or df_to_job.get(df, None) or cur_job
        proj = job_to_project.get(job, proj or df_to_project.get(df, cur_proj))
        return proj or "", job or "", df or ""

    rows=[]

    # ---------- A) TRANSFORMATION (expression) columns from DIElement ----------
    for e in root.iter():
        if not isinstance(e.tag, str): continue
        tag = lower(strip_ns(e.tag))
        a   = attrs_ci(e)

        if tag in PROJECT_TAGS: cur_proj=(a.get("name") or a.get("displayname") or cur_proj).strip()
        if tag in DF_TAGS:      cur_df  =(a.get("name") or a.get("displayname") or cur_df).strip()
        if tag in JOB_TAGS:     cur_job = _job_name_from_node(e) or (a.get("name") or a.get("displayname") or cur_job).strip()

        if tag == "dielement" and in_dataflow(e, pm):
            proj, job, df = context_for(e)
            col_name = (a.get("name") or "").strip()

            # mapping text (ui_mapping_text)
            map_txt = ""
            for ch in e.iter():
                if lower(strip_ns(getattr(ch,"tag",""))) == "diattribute" and \
                   lower(getattr(ch, "attrib", {}).get("name","")) == "ui_mapping_text":
                    map_txt = (getattr(ch, "attrib", {}).get("value") or "").strip()
                    if map_txt: break
            if map_txt:
                map_txt = html.unescape(map_txt).replace("\r", " ").replace("\n", " ").strip()

            transform = nearest_schema_name(e, pm)

            # build row (TRANSFORM / column mapping)
            row = {
                "PROJECT_NAME": proj,
                "JOB_NAME": job,
                "DATAFLOW_NAME": df,
                "TRANSFORMATION_NAME": transform,
                "TRANSFORMATION_TYPE": "TRANSFORM",  # <-- TRANSFORMATION append
                "COLUMN_NAME": col_name,
                "MAPPING_TEXT": map_txt,
            }
            emit(rows, **row)  # append TRANSFORM row

    # ---------- B) SOURCES & TARGETS ----------
    table_catalog = extract_ditable_catalog(root)    # global DITables (3-key)
    st_refs       = collect_source_target_refs(root) # per-DF references

    for r in st_refs:
        # fill project/job context by DF
        job  = df_to_job.get(r["df"], "") or ""
        proj = job_to_project.get(job, "") or df_to_project.get(r["df"], "")

        if r["kind"] == "DATABASE":
            key = (r["dsC"], r["ownerC"], r["tableC"])
            cols = table_catalog.get(key, set())
            for col in sorted(cols):
                # build row (DATABASE source/target → one row per column)
                row = {
                    "PROJECT_NAME": proj,
                    "JOB_NAME": job,
                    "DATAFLOW_NAME": r["df"],
                    "TRANSFORMATION_NAME": r["schema"],
                    "TRANSFORMATION_TYPE": r["role"],  # <-- DB SOURCE/TARGET append
                    "COLUMN_NAME": col,
                    "MAPPING_TEXT": "",
                }
                emit(rows, **row)  # append DB source/target row
        else:
            # FILE or EXCEL: no global columns; emit a single row with blank COLUMN_NAME
            row = {
                "PROJECT_NAME": proj,
                "JOB_NAME": job,
                "DATAFLOW_NAME": r["df"],
                "TRANSFORMATION_NAME": r["schema"],
                "TRANSFORMATION_TYPE": r["role"],  # <-- FILE/EXCEL SOURCE/TARGET append
                "COLUMN_NAME": "",
                "MAPPING_TEXT": "",
            }
            emit(rows, **row)  # append FILE/EXCEL row

    # finalize dataframe
    if rows:
        df = pd.DataFrame(rows, columns=[
            "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
            "TRANSFORMATION_NAME","TRANSFORMATION_TYPE",
            "COLUMN_NAME","MAPPING_TEXT"
        ])
        # optional dedupe for safety
        df["__k__"] = df.astype(str).agg("||".join, axis=1)
        df = df.drop_duplicates("__k__").drop(columns="__k__").reset_index(drop=True)
        return df

    return pd.DataFrame(columns=[
        "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
        "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","COLUMN_NAME","MAPPING_TEXT"
    ])

# -------------------- main (v3-style) --------------------
def main():
    now = datetime.datetime.now()
    ts  = now.strftime("%Y%m%d_%H%M%S")

    path = r"C:\Users\raksahu\Downloads\python\input\sap_ds_xml_files"
    single_file = fr"{path}\export_af.xml"   # optional pin

    all_files = glob.glob(os.path.join(path, "*.xml"))
    if single_file in all_files:
        all_files = [single_file]

    print(f"total number of files present in the path ({len(all_files)})")
    print(all_files)

    frames=[]
    for i, file in enumerate(all_files):
        print(f"Row Number:{i}--{file}")
        frames.append(parse_single_xml(file))

    final_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
                 "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","COLUMN_NAME","MAPPING_TEXT"]
    )

    out_base = fr"{path}\SAPDS_ALL_COLUMN_MAPPING_{ts}"
    csv_path = out_base + ".csv"
    xlsx_path = out_base + ".xlsx"

    final_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        final_df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"Excel write skipped: {e}")

    print(f"Done. Wrote: {csv_path} | Rows: {len(final_df)}")

if __name__ == "__main__":
    main()
