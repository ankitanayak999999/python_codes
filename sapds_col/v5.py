#08/28 - ALL TARNSAFORMATION WORKING FILE ,SOURCE COLUMNS ADDED , SOURCE AND TARGET ROWS ADDED , FILES ROWS ADDED 
#MISSING TRANSAFORMATION NAME FOR SOURCE ,TARGET ANF FILE ROWS 

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
    row = dict(kwargs)   # build row
    rows.append(row)     # append row

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

# -------------------- context helpers (same logic you use elsewhere) --------------------
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
# 1) GLOBAL DITABLE CATALOG (strict 3‑key)
# ======================================================================
def extract_ditable_catalog(root):
    """
    Build a catalog from global <DITable>:
      (DATASTORE, OWNER, TABLE) -> set(COLUMN_NAME)
    Keys are canonicalized with _canon().
    """
    by3 = defaultdict(set)
    for tbl in root.iter():
        if lower(strip_ns(getattr(tbl, "tag", ""))) != "ditable":
            continue
        a = attrs_ci(tbl)
        ds   = _canon(a.get("datastore")    or a.get("datastorename") or "")
        owner= _canon(a.get("owner")        or a.get("ownername")     or "")
        tnam = _canon(a.get("name")         or a.get("tablename")     or a.get("table_name") or "")
        if not (ds and owner and tnam):
            continue  # strict 3-key: skip incomplete

        cols=set()
        for c in tbl.iter():
            if lower(strip_ns(getattr(c,"tag",""))) == "dicolumn":
                cn = (attrs_ci(c).get("name") or "").strip()
                if cn: cols.add(_canon(cn))
        if cols:
            by3[(ds, owner, tnam)] |= cols
    return by3

# ======================================================================
# 2) PER‑DF SOURCE/TARGET REFERENCES (DB / FILE / EXCEL)
# ======================================================================
def collect_source_target_refs(root):
    """
    Walks the tree and collects any Source/Target nodes, with DF & Schema context.
    Returns: list of dicts (df, schema, role, kind, dsC/ownerC/tableC for DB).
    """
    pm = build_parent_map(root)
    refs=[]
    for n in root.iter():
        tag=lower(strip_ns(getattr(n,"tag","")))
        if tag not in ALL_ST_TAGS:
            continue
        a=attrs_ci(n)

        # backtrack to find DF & DISchema
        df_name=schema_name=""
        for anc in ancestors(n, pm, 200):
            t=lower(strip_ns(getattr(anc,"tag","")))
            at=attrs_ci(anc)
            if not df_name and t in DF_TAGS:
                df_name=(at.get("name") or at.get("displayname") or "").strip()
            if not schema_name and t=="dischema":
                schema_name=(at.get("name") or at.get("displayname") or "").strip()

        role = "SOURCE" if tag in ALL_SOURCE_TAGS else "TARGET"
        if   tag in DB_SOURCE_TAGS + DB_TARGET_TAGS:   kind="DATABASE"
        elif tag in FILE_SOURCE_TAGS + FILE_TARGET_TAGS: kind="FILE"
        elif tag in XLS_SOURCE_TAGS  + XLS_TARGET_TAGS:  kind="EXCEL"
        else: kind="UNKNOWN"

        refs.append({
            "df": df_name,
            "schema": schema_name,
            "role": role,
            "kind": kind,
            "dsC": _canon(a.get("datastorename") or a.get("datastore") or ""),
            "ownerC": _canon(a.get("ownername") or a.get("owner") or a.get("schema") or ""),
            "tableC": _canon(a.get("tablename") or a.get("table") or a.get("name") or ""),
        })
    return refs
#helper to extract source columns from mapping text
 def _extract_source_columns(mapping_text: str) -> str:
    if not mapping_text:
    return ""
    txt = str(mapping_text)
    # remove inline tokens starting with '#', e.g. "#NULL" or "#CLM"
    txt = re.sub(r"#\S+", " ", txt)
    matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\b", txt)
    seen, ordered = set(), []
    for m in matches:
    key = m.upper()
    if key not in seen:
    seen.add(key)
    ordered.append(m)
    return ",".join(ordered)

# ======================================================================
# 3) HELPERS TO EMIT ROWS (each can be toggled independently)
# ======================================================================
def collect_transform_rows(root, pm, context_for, rows):
    """Append TRANSFORM rows from DIElement (inside a DF)."""
    for e in root.iter():
        if not isinstance(e.tag, str): continue
        tag = lower(strip_ns(e.tag))
        a   = attrs_ci(e)

        if tag == "dielement" and in_dataflow(e, pm):
            proj, job, df = context_for(e)
            col_name = (a.get("name") or "").strip()

            # ui_mapping_text (mapping expression)
            map_txt = ""
            for ch in e.iter():
                if lower(strip_ns(getattr(ch,"tag",""))) == "diattribute" and \
                   lower(getattr(ch, "attrib", {}).get("name","")) == "ui_mapping_text":
                    map_txt = (getattr(ch, "attrib", {}).get("value") or "").strip()
                    if map_txt: break
            if map_txt:
                map_txt = html.unescape(map_txt).replace("\r", " ").replace("\n", " ").strip()

            transform = nearest_schema_name(e, pm)
            src_cols=_extract_source_columns(map_txt)

            # ---------- TRANSFORMATION ROW ----------
            row = {
                "PROJECT_NAME": proj,
                "JOB_NAME": job,
                "DATAFLOW_NAME": df,
                "TRANSFORMATION_NAME": transform,
                "TRANSFORMATION_TYPE": "TRANSFORM",      # TRANSFORMATION append
                "TARGET_COLUMN": col_name,
                "SOURCE_COLUMNS":src_cols,
                "MAPPING_TEXT": map_txt,
            }
            emit(rows, **row)  # append TRANSFORM row

def collect_db_st_rows(root, df_to_job, job_to_project, df_to_project, rows):
    """Append DATABASE SOURCE/TARGET rows (one row per column via <DITable>)."""
    table_catalog = extract_ditable_catalog(root)
    st_refs       = collect_source_target_refs(root)

    for r in st_refs:
        if r["kind"] != "DATABASE":
            continue

        # fill context by DF
        job  = df_to_job.get(r["df"], "") or ""
        proj = job_to_project.get(job, "") or df_to_project.get(r["df"], "")

        key = (r["dsC"], r["ownerC"], r["tableC"])
        cols = table_catalog.get(key, set())
        for col in sorted(cols):
            # ---------- DATABASE SOURCE/TARGET ROW ----------
            row = {
                "PROJECT_NAME": proj,
                "JOB_NAME": job,
                "DATAFLOW_NAME": r["df"],
                "TRANSFORMATION_NAME": r["schema"],
                "TRANSFORMATION_TYPE": r["role"],        # DB SOURCE/TARGET append
                "TARGET_COLUMN": col,
                "SOURCE_COLUMNS":"",
                "MAPPING_TEXT": "",
            }
            emit(rows, **row)  # append DB source/target row

def collect_file_excel_st_rows(root, df_to_job, job_to_project, df_to_project, rows):
    """Append FILE/EXCEL SOURCE/TARGET rows (no column list)."""
    st_refs = collect_source_target_refs(root)
    for r in st_refs:
        if r["kind"] not in ("FILE","EXCEL"):
            continue

        job  = df_to_job.get(r["df"], "") or ""
        proj = job_to_project.get(job, "") or df_to_project.get(r["df"], "")

        # ---------- FILE / EXCEL SOURCE/TARGET ROW ----------
        row = {
            "PROJECT_NAME": proj,
            "JOB_NAME": job,
            "DATAFLOW_NAME": r["df"],
            "TRANSFORMATION_NAME": r["schema"],
            "TRANSFORMATION_TYPE": r["role"],           # FILE/EXCEL append
            "TARGET_COLUMN": "",
            "SOURCE_COLUMNS":"",
            "MAPPING_TEXT": "",
        }
        emit(rows, **row)  # append FILE/EXCEL row

# ======================================================================
# 4) MAIN PARSER
# ======================================================================
def parse_single_xml(xml_path: str) -> pd.DataFrame:
    parser = ET.XMLParser(huge_tree=True, recover=True)
    root   = ET.parse(xml_path, parser=parser).getroot()
    pm     = build_parent_map(root)

    # Maps used to resolve PROJECT/JOB for a DF (same as your other script)
    job_to_project = build_job_to_project_map(root)
    df_to_project  = build_df_project_map(root)
    df_to_job      = build_df_job_map(root)

    # context resolver used by TRANSFORM helper
    def context_for(e):
        proj=job=df=""
        for a in ancestors(e, pm, 200):
            t  = lower(strip_ns(getattr(a,"tag","")))
            at = attrs_ci(a)
            nm = (at.get("name") or at.get("displayname") or "").strip()
            if not df and t in DF_TAGS: df = nm or df
            if not proj and t in PROJECT_TAGS: proj = nm or proj
            if t in JOB_TAGS and not job: job = _job_name_from_node(a) or job
        df  = df or ""
        job = job or df_to_job.get(df, None) or ""
        proj = job_to_project.get(job, proj or df_to_project.get(df, ""))
        return proj or "", job or "", df or ""

    rows = []

    # ---- call helpers (each on its own line; easy to toggle) ----
    collect_transform_rows(root, pm, context_for, rows)                         # TRANSFORM rows
    collect_db_st_rows(root, df_to_job, job_to_project, df_to_project, rows)    # DB SOURCE/TARGET rows
    collect_file_excel_st_rows(root, df_to_job, job_to_project, df_to_project, rows)  # FILE/EXCEL rows

    # finalize dataframe
    if rows:
        df = pd.DataFrame(rows, columns=[
            "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
            "TRANSFORMATION_NAME","TRANSFORMATION_TYPE",
            "TARGET_COLUMN","SOURCE_COLUMNS","MAPPING_TEXT"
        ])
        # (no rename here—done in main to keep your pattern)
        df["__k__"] = df.astype(str).agg("||".join, axis=1)
        df = df.drop_duplicates("__k__").drop(columns="__k__").reset_index(drop=True)
        return df
    empty_df=pd.DataFrame(columns=[
        "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
        "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","TARGET_COLUMN","SOURCE_COLUMNS","MAPPING_TEXT"])
    return empty_df


# ======================================================================
# 5) main()  — SAME LAYOUT AS YOUR V4 (only added new col + rename here)
# ======================================================================
def main():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # keep your paths here (unchanged)
    path = r"C:\Users\raksahu\Downloads\python\input\sap_ds_xml_files"
    single_file = fr"{path}\export_af.xml"   # optional pin

    all_files = glob.glob(os.path.join(path, "*.xml"))
    if single_file in all_files:
        all_files = [single_file]

    print(f"total number of files present in the path ({len(all_files)})")
    print(all_files)

    out_frames=[]
    for i, file in enumerate(all_files):
        print(f"Row Number:{i}--{file}")
        out_frames.append(parse_single_xml(file))

    final_df = pd.concat(out_frames, ignore_index=True)

    #RENAME 
    rename_mapping = {
        'PROJECT_NAME'       : 'PROJECT_NAME',
        'JOB_NAME'           : 'JOB_NAME',
        'DATAFLOW_NAME'      : 'DATAFLOW_NAME',
        'TRANSFORMATION_NAME': 'TRANSFORMATION_NAME',
        'TRANSFORMATION_TYPE': 'TRANSFORMATION_TYPE',
        'TARGET_COLUMN'      : 'TARGET_COLUMN',
        'SOURCE_COLUMNS'     : 'SOURCE_COLUMNS',
        'MAPPING_TEXT'       : 'MAPPING_TEXT',
    }
    final_df = final_df.rename(columns=rename_mapping)[list(rename_mapping.values())]

    # RECORD_KEY for duplicate snapshot (same structure you showed)
    key_cols = ["PROJECT_NAME","JOB_NAME","DATAFLOW_NAME","TRANSFORMATION_NAME","TARGET_COLUMN"]
    final_df["RECORD_KEY"] = final_df[key_cols].astype(str).agg("|".join, axis=1)

    dups_df = final_df[final_df.duplicated(subset=["RECORD_KEY"], keep=False)].copy()
    dups_df["DUP_GROUP"] = dups_df.groupby("RECORD_KEY").ngroup() + 1
    dups_df["DUP_COUNT"] = dups_df.groupby("RECORD_KEY")["RECORD_KEY"].transform("count")

    output_path = fr"{path}\SAP_DS_ALL_TABLE_MAPPING_{timestamp}.csv"
    dups_path   = fr"{path}\SAP_DS_TABLE_MAPPING_DUPLICATES_{timestamp}.csv"

    final_df.to_csv(output_path, index=False)
    dups_df.to_csv(dups_path, index=False)

    print(f"Done. Wrote: {output_path} | Rows: {len(final_df)}")

if __name__ == "__main__":
    main()
