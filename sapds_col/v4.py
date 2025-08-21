# sapds_column_mappings_v3_plus_st_helpers.py
# v3 + helpers to collect global DITables and per-DF Source/Target references
# Strict 3-key link (DATASTORE, OWNER, TABLE) for database sources/targets

import re, os, glob, html, datetime
import pandas as pd
from lxml import etree as ET
from collections import defaultdict

# -------------------- small utils (unchanged) --------------------
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

# ---------- tag sets (as before) ----------
DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")
CALLSTEP_TAGS = ("dicallstep","callstep")

# Source/Target containers we care about under schemas
DB_SOURCE_TAGS   = ("didatabasetablesource","ditablesource")
DB_TARGET_TAGS   = ("didatabasetabletarget","ditabletarget")
FILE_SOURCE_TAGS = ("difilesource",)
FILE_TARGET_TAGS = ("difiletarget",)
XLS_SOURCE_TAGS  = ("diexcelsource",)
XLS_TARGET_TAGS  = ("diexceltarget",)

ALL_SOURCE_TAGS = DB_SOURCE_TAGS + FILE_SOURCE_TAGS + XLS_SOURCE_TAGS
ALL_TARGET_TAGS = DB_TARGET_TAGS + FILE_TARGET_TAGS + XLS_TARGET_TAGS
ALL_ST_TAGS     = ALL_SOURCE_TAGS + ALL_TARGET_TAGS

# -------------------- v3 context helpers (unchanged) --------------------
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

    Notes:
      - Keys are *cleaned* with _canon() (uppercased, trimmed, whitespace collapsed).
      - We only keep entries where all three keys are present.
      - This catalog is used ONLY for Database Source/Target expansion.
    """
    by3 = defaultdict(set)

    for tbl in root.iter():
        if lower(strip_ns(getattr(tbl, "tag", ""))) != "ditable":
            continue
        a = attrs_ci(tbl)

        # Accept common attribute spellings seen in exports
        ds   = a.get("datastore") or a.get("datastorename") or ""
        owner= a.get("owner")     or a.get("ownername")     or ""
        tnam = a.get("name")      or a.get("tablename")     or a.get("table_name") or ""

        dsC, ownerC, tC = _canon(ds), _canon(owner), _canon(tnam)
        if not (dsC and ownerC and tC):
            continue  # strict: require all 3

        # Collect child <DIColumn name="...">
        cols = set()
        for c in tbl.iter():
            if lower(strip_ns(getattr(c, "tag", ""))) == "dicolumn":
                cn = (attrs_ci(c).get("name") or "").strip()
                if cn:
                    cols.add(cn)

        if cols:
            by3[(dsC, ownerC, tC)] |= cols

    return by3

# ======================================================================
# 2) PER-DF SOURCE/TARGET REFERENCES
# ======================================================================
def collect_source_target_refs(root):
    """
    Walk each Dataflow -> DISchema and capture Source/Target definitions.

    For each schema returns a record:
      {
        'df': <dataflow name>,
        'schema': <schema name>,
        'role': 'SOURCE' | 'TARGET',
        'kind': 'DATABASE' | 'FILE' | 'EXCEL',
        # cleaned link key pieces (for DATABASE only)
        'dsC': <canon datastore>,
        'ownerC': <canon owner>,
        'tableC': <canon table>,
        # raw-ish identifiers (useful to display / name the object)
        'datastore': <raw>,
        'owner_or_format': <owner/schema for DB or format for file/excel>,
        'object_name': <table/file/sheet/outview>,
      }

    Linking rule:
      - DATABASE: link with strict 3-key (dsC, ownerC, tableC) into the DITable catalog.
      - FILE/EXCEL: no global catalog; we just report the object (no column enumeration).
    """
    refs = []

    for df_node in root.iter():
        if lower(strip_ns(getattr(df_node, "tag", ""))) not in DF_TAGS:
            continue
        df_name = (attrs_ci(df_node).get("name") or attrs_ci(df_node).get("displayname") or "").strip()
        if not df_name:
            continue

        for sc in df_node.iter():
            if lower(strip_ns(getattr(sc, "tag", ""))) != "dischema":
                continue
            sch_name = (attrs_ci(sc).get("name") or attrs_ci(sc).get("displayname") or "").strip()
            if not sch_name:
                continue

            # prefer DIOutputView as a logical name (esp. for file/excel)
            outview = ""
            for ch in sc.iter():
                if lower(strip_ns(getattr(ch, "tag", ""))) == "dioutputview":
                    outview = (attrs_ci(ch).get("name") or "").strip()
                    if outview:
                        break

            # Look under this schema for any ST containers
            for ch in sc.iter():
                tag = lower(strip_ns(getattr(ch, "tag", "")))
                if tag not in ALL_ST_TAGS:
                    continue
                a = attrs_ci(ch)

                # Decide role + kind
                if tag in DB_SOURCE_TAGS:   role, kind = "SOURCE", "DATABASE"
                elif tag in DB_TARGET_TAGS: role, kind = "TARGET", "DATABASE"
                elif tag in FILE_SOURCE_TAGS: role, kind = "SOURCE", "FILE"
                elif tag in FILE_TARGET_TAGS: role, kind = "TARGET", "FILE"
                elif tag in XLS_SOURCE_TAGS:  role, kind = "SOURCE", "EXCEL"
                else:                          role, kind = "TARGET", "EXCEL"

                if kind == "DATABASE":
                    ds   = a.get("datastorename") or a.get("datastore") or ""
                    owner= a.get("ownername")     or a.get("owner")     or a.get("schema") or ""
                    tbl  = a.get("tablename")     or a.get("table")     or a.get("name")   or ""
                    refs.append({
                        "df": df_name,
                        "schema": sch_name,
                        "role": role,
                        "kind": kind,
                        "dsC": _canon(ds),
                        "ownerC": _canon(owner),
                        "tableC": _canon(tbl),
                        "datastore": ds,
                        "owner_or_format": owner,
                        "object_name": tbl or outview or "",
                    })
                elif kind in ("FILE","EXCEL"):
                    fmt  = a.get("formatname") or a.get("file_format") or kind
                    # best-effort object name: filename/dataset/outview
                    obj  = a.get("filename") or a.get("datasetname") or outview or ""
                    ds   = a.get("datastorename") or a.get("datastore") or kind
                    refs.append({
                        "df": df_name,
                        "schema": sch_name,
                        "role": role,
                        "kind": kind,
                        "dsC": "", "ownerC": "", "tableC": "",
                        "datastore": ds,
                        "owner_or_format": fmt,
                        "object_name": obj,
                    })
    return refs

# ======================================================================
# 3) MAIN COLUMN MAPPING (v3 + ST expansion)
# ======================================================================
def parse_single_xml(xml_path: str) -> pd.DataFrame:
    parser = ET.XMLParser(huge_tree=True, recover=True)
    root   = ET.parse(xml_path, parser=parser).getroot()
    pm     = build_parent_map(root)

    # v3 context
    job_to_project = build_job_to_project_map(root)
    df_to_project  = build_df_project_map(root)
    df_to_job      = build_df_job_map(root)

    cur_proj = cur_job = cur_df = ""

    def context_for(e):
        proj=job=df=""
        for a in ancestors(e, pm):
            t  = lower(strip_ns(a.tag))
            at = attrs_ci(a)
            nm = (at.get("name") or at.get("displayname") or "").strip()
            if not df and t in DF_TAGS: df = nm or df
            if not proj and t in PROJECT_TAGS: proj = nm or proj
            if t in JOB_TAGS and not job: job = _job_name_from_node(a) or job
        df  = df or cur_df
        job = job or df_to_job.get(df, None) or (cur_job)
        proj = job_to_project.get(job, proj or df_to_project.get(df, cur_proj))
        return proj or "", job or "", df or ""

    rows = []

    # --- A) TRANSFORM columns from DIElement (original v3 logic) ---
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

            rows.append({
                "PROJECT_NAME": proj,
                "JOB_NAME": job,
                "DATAFLOW_NAME": df,
                "TRANSFORMATION_NAME": transform,
                "TRANSFORMATION_TYPE": "TRANSFORM",
                "COLUMN_NAME": col_name,
                "MAPPING_TEXT": map_txt
            })

    # --- B) SOURCE/TARGET expansion using helpers ---
    # 1) global DITables (strict 3-key)
    table_catalog = extract_ditable_catalog(root)

    # 2) per-DF ST references
    st_refs = collect_source_target_refs(root)

    # 3) expand
    for r in st_refs:
        proj = job = ""
        # context by DF name
        job = df_to_job.get(r["df"], "") or ""
        proj = job_to_project.get(job, "") or df_to_project.get(r["df"], "")

        if r["kind"] == "DATABASE":
            key = (r["dsC"], r["ownerC"], r["tableC"])
            cols = table_catalog.get(key, set())
            # strict mode: only emit when catalog is present
            for col in sorted(cols):
                rows.append({
                    "PROJECT_NAME": proj,
                    "JOB_NAME": job,
                    "DATAFLOW_NAME": r["df"],
                    "TRANSFORMATION_NAME": r["schema"],  # schema node name
                    "TRANSFORMATION_TYPE": r["role"],     # SOURCE or TARGET
                    "COLUMN_NAME": col,
                    "MAPPING_TEXT": ""
                })
        else:
            # FILE/EXCEL: no catalog => one row with blank COLUMN_NAME (as agreed)
            rows.append({
                "PROJECT_NAME": proj,
                "JOB_NAME": job,
                "DATAFLOW_NAME": r["df"],
                "TRANSFORMATION_NAME": r["schema"],
                "TRANSFORMATION_TYPE": r["role"],
                "COLUMN_NAME": "",
                "MAPPING_TEXT": ""
            })

    # --- finish ---
    if rows:
        df = pd.DataFrame(rows)
        # order & dedupe
        keep_cols = [
            "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
            "TRANSFORMATION_NAME","TRANSFORMATION_TYPE",
            "COLUMN_NAME","MAPPING_TEXT"
        ]
        df = df[keep_cols]
        df["__k__"] = df.astype(str).agg("||".join, axis=1)
        df = df.drop_duplicates(subset="__k__").drop(columns="__k__").reset_index(drop=True)
        return df

    return pd.DataFrame(columns=[
        "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
        "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","COLUMN_NAME","MAPPING_TEXT"
    ])

# -------------------- main (same shape as your v3) --------------------
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

    out_frames=[]
    for i, file in enumerate(all_files):
        print(f"Row Number:{i}--{file}")
        out_frames.append(parse_single_xml(file))

    final_df = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame(
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
