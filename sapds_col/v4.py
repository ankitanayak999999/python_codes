# sapds_column_mapping_v3.py
# Column-level extractor for SAP DS XML exports
# Emits SOURCE / TRANSFORM / TARGET rows with schema order and mapping text.

import re, os, glob, html, datetime
import pandas as pd
from lxml import etree as ET
from collections import defaultdict

# ---------------- utils ----------------
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
def norm(s): return re.sub(r"\s+", " ", (s or "").strip()).upper()

# ---------------- tag sets ----------------
DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")
CALLSTEP_TAGS = ("dicallstep","callstep")

# include the container tags seen in your export
SOURCE_CONTAINER_TAGS = (
    "didatabasetablesource", "difilesource", "diexcelsource",
    "ditablesource"  # <DITableSource>
)
TARGET_CONTAINER_TAGS = (
    "didatabasetabletarget", "difiletarget", "diexceltarget",
    "ditabletarget"  # <DITableTarget>
)

# ---------------- context helpers ----------------
def _job_name_from_node(job_node):
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch, "tag", ""))) == "diattribute" and lower(ch.attrib.get("name", "")) == "job_name":
            v = (ch.attrib.get("value") or "").strip()
            if v: return v
    return (job_node.attrib.get("name") or job_node.attrib.get("displayName") or "").strip()

def collect_df_names(root):
    out=set()
    for n in root.iter():
        if lower(strip_ns(getattr(n,"tag",""))) in DF_TAGS:
            nm=(n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: out.add(nm)
    return out

def build_job_to_project_map(root):
    j2p={}
    for p in root.iter():
        if lower(strip_ns(getattr(p,"tag",""))) not in PROJECT_TAGS: continue
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
    if len(projects) == 1:
        only = projects[0][0]
        for dn in df_names: df_proj.setdefault(dn, only)
    return df_proj

def build_df_job_map(root):
    # minimal but safe: if only one job, map all DFs to it
    jobs = []
    for n in root.iter():
        if lower(strip_ns(getattr(n, "tag", ""))) in JOB_TAGS:
            nm = _job_name_from_node(n)
            if nm: jobs.append(nm)
    df_names = collect_df_names(root)
    df_job = {}
    if len(jobs) == 1:
        only = jobs[0]
        for d in df_names: df_job[d] = only
    return df_job

def in_dataflow(e, pm):
    for a in ancestors(e, pm, 200):
        if lower(strip_ns(getattr(a, "tag", ""))) in DF_TAGS:
            return True
    return False

def nearest_schema_name(e, pm):
    for a in ancestors(e, pm, 200):
        if lower(strip_ns(getattr(a, "tag", ""))) == "dischema":
            nm = (attrs_ci(a).get("name") or attrs_ci(a).get("displayname") or "").strip()
            if nm: return nm
    return ""

# ---------------- global <DITable> catalog ----------------
def build_table_catalog(root):
    """
    Build {(DS, OWNER, TABLE) -> {column,...}} from global <DITable> blocks.
    """
    catalog = defaultdict(set)
    for n in root.iter():
        if lower(strip_ns(getattr(n, "tag", ""))) != "ditable":
            continue
        a = attrs_ci(n)
        tbl   = (a.get("name") or a.get("table_name") or "").strip()
        owner = (a.get("owner") or a.get("ownername") or "").strip()
        ds    = (a.get("datastore") or a.get("datastorename") or "").strip()
        if not (tbl and ds): 
            continue
        key = (norm(ds), norm(owner), norm(tbl))
        for col in n.iter():
            if lower(strip_ns(getattr(col, "tag", ""))) == "dicolumn":
                cn = (attrs_ci(col).get("name") or "").strip()
                if cn:
                    catalog[key].add(cn)
    return catalog

# ---------------- per-DF graph: levels, explicit cols, db refs, input views ----------------
def compute_df_structures(root):
    """
    Returns:
      df_levels:      {df: {schema: level}}
      df_cols:        {df: {schema: set(cols)}}     # DIElements seen in that schema
      df_dbref:       {df: {schema: (role, ds, owner, table)}}  # SOURCE/TARGET ref if present
      df_inp_views:   {df: {schema: set(input_schema_names)}}   # DIInputView names under schema
    """
    df_levels    = {}
    df_cols      = defaultdict(lambda: defaultdict(set))
    df_dbref     = defaultdict(dict)
    df_inp_views = defaultdict(lambda: defaultdict(set))

    for df_node in root.iter():
        if lower(strip_ns(getattr(df_node, "tag", ""))) not in DF_TAGS:
            continue
        df_name = (attrs_ci(df_node).get("name") or attrs_ci(df_node).get("displayname") or "").strip()
        if not df_name: 
            continue

        levels = {}
        next_level = 1
        for sc in df_node.iter():
            if lower(strip_ns(getattr(sc, "tag", ""))) != "dischema":
                continue
            sch = (attrs_ci(sc).get("name") or attrs_ci(sc).get("displayname") or "").strip()
            if not sch:
                continue
            if sch not in levels:
                levels[sch] = next_level
                next_level += 1

            # collect DIElements (explicit transform columns)
            for el in sc.iter():
                if lower(strip_ns(getattr(el, "tag", ""))) == "dielement":
                    cn = (attrs_ci(el).get("name") or "").strip()
                    if cn:
                        df_cols[df_name][sch].add(cn)

            # record DIInputView(s) (actual upstream schema names)
            for iv in sc.iter():
                if lower(strip_ns(getattr(iv, "tag", ""))) in ("diinputview","inputview"):
                    inp = (attrs_ci(iv).get("name") or "").strip()
                    if inp:
                        df_inp_views[df_name][sch].add(inp)

            # find a Source/Target container & pull its db reference
            for ch in sc.iter():
                t = lower(strip_ns(getattr(ch, "tag", "")))
                if t in SOURCE_CONTAINER_TAGS or t in TARGET_CONTAINER_TAGS or "source" in t or "target" in t:
                    a = attrs_ci(ch)
                    ds    = (a.get("datastorename") or a.get("datastore") or "").strip()
                    owner = (a.get("ownername")     or a.get("owner")     or a.get("schema") or "").strip()
                    tbl   = (a.get("tablename")     or a.get("table")     or a.get("name")   or "").strip()
                    role  = "SOURCE" if t in SOURCE_CONTAINER_TAGS or "source" in t else "TARGET"
                    if ds and tbl:
                        df_dbref[df_name][sch] = (role, ds, owner, tbl)
                        break  # one ref is enough

        if levels:
            df_levels[df_name] = levels

    return df_levels, df_cols, df_dbref, df_inp_views

def transformation_type_by_tags(e, pm):
    for a in ancestors(e, pm, 200):
        t = lower(strip_ns(getattr(a, "tag", "")))
        if t in SOURCE_CONTAINER_TAGS: return "SOURCE"
        if t in TARGET_CONTAINER_TAGS: return "TARGET"
        if t == "dischema": break
    # default for DIElements living in transform/query schemas
    return "TRANSFORM"

# ---------------- parse one xml ----------------
def parse_single_xml(xml_path: str) -> pd.DataFrame:
    parser = ET.XMLParser(huge_tree=True, recover=True)
    root   = ET.parse(xml_path, parser=parser).getroot()
    pm     = build_parent_map(root)

    # Context maps
    job_to_project = build_job_to_project_map(root)
    df_to_project  = build_df_project_map(root)
    df_to_job      = build_df_job_map(root)

    # Structures
    table_catalog = build_table_catalog(root)
    df_levels, df_cols, df_dbref, df_inp_views = compute_df_structures(root)

    # df -> (proj, job) context
    df_context = {}
    for df in df_levels.keys():
        job = df_to_job.get(df, "") or ""
        proj = job_to_project.get(job, "") or df_to_project.get(df, "")
        df_context[df] = (proj, job)

    rows = []

    # Pass 1: explicit DIElement rows (TRANSFORM and any S/T that enumerate DIElements)
    for e in root.iter():
        if not isinstance(e.tag, str): continue
        tag = lower(strip_ns(e.tag))
        if tag != "dielement" or not in_dataflow(e, pm): 
            continue

        a = attrs_ci(e)
        col_name = (a.get("name") or "").strip()
        transform = nearest_schema_name(e, pm)

        # find DF for this element
        df_name = ""
        for aup in ancestors(e, pm, 200):
            if lower(strip_ns(getattr(aup,"tag",""))) in DF_TAGS:
                df_name = (attrs_ci(aup).get("name") or attrs_ci(aup).get("displayname") or "").strip()
                break

        if not df_name: 
            continue

        # mapping text
        map_txt = ""
        for ch in e.iter():
            if lower(strip_ns(getattr(ch, "tag", ""))) == "diattribute" and \
               lower(getattr(ch, "attrib", {}).get("name", "")) == "ui_mapping_text":
                map_txt = (getattr(ch, "attrib", {}).get("value") or "").strip()
                if map_txt: break
        if map_txt:
            map_txt = html.unescape(map_txt).replace("\r", " ").replace("\n", " ").strip()

        t_type = transformation_type_by_tags(e, pm)
        lvl = df_levels.get(df_name, {}).get(transform, 1)
        proj, job = df_context.get(df_name, ("",""))

        rows.append({
            "PROJECT_NAME": proj,
            "JOB_NAME": job,
            "DATAFLOW_NAME": df_name,
            "TRANSFORMATION_NAME": transform,
            "TRANSFORMATION_TYPE": t_type,
            "TRANSFORMATION_LEVEL": int(lvl),
            "COLUMN_NAME": col_name,
            "MAPPING_TEXT": map_txt
        })
        if transform and col_name:
            df_cols[df_name][transform].add(col_name)

    # Pass 2: add SOURCE/TARGET schema columns using db refs + catalog; fallback to DIInputView for TARGET
    for df_name, schemas in df_levels.items():
        proj, job = df_context.get(df_name, ("",""))
        for sch_name, lvl in schemas.items():
            ref = df_dbref.get(df_name, {}).get(sch_name)
            if not ref:
                continue
            role, ds, owner, tbl = ref
            key = (norm(ds), norm(owner), norm(tbl))
            cols = set(table_catalog.get(key, set()))

            # Fallback for TARGET: copy from its input views (union of upstream schema columns)
            if not cols and role == "TARGET":
                inp_views = df_inp_views.get(df_name, {}).get(sch_name, set())
                for iv in inp_views:
                    cols |= df_cols.get(df_name, {}).get(iv, set())

            # If still nothing for SOURCE, that's acceptable (you said Source can be blank)
            if not cols:
                continue

            for col_name in sorted(cols):
                rows.append({
                    "PROJECT_NAME": proj,
                    "JOB_NAME": job,
                    "DATAFLOW_NAME": df_name,
                    "TRANSFORMATION_NAME": sch_name,
                    "TRANSFORMATION_TYPE": role,                # SOURCE / TARGET
                    "TRANSFORMATION_LEVEL": int(lvl),
                    "COLUMN_NAME": col_name,
                    "MAPPING_TEXT": ""                           # per requirement
                })

    # Deduplicate identical rows
    if rows:
        df = pd.DataFrame(rows)
        df["__k__"] = (
            df["PROJECT_NAME"].astype(str) + "||" +
            df["JOB_NAME"].astype(str) + "||" +
            df["DATAFLOW_NAME"].astype(str) + "||" +
            df["TRANSFORMATION_NAME"].astype(str) + "||" +
            df["TRANSFORMATION_TYPE"].astype(str) + "||" +
            df["COLUMN_NAME"].astype(str) + "||" +
            df["MAPPING_TEXT"].astype(str)
        )
        df = df.drop_duplicates(subset="__k__").drop(columns="__k__")
        return df[[
            "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
            "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","TRANSFORMATION_LEVEL",
            "COLUMN_NAME","MAPPING_TEXT"
        ]].reset_index(drop=True)

    return pd.DataFrame(columns=[
        "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
        "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","TRANSFORMATION_LEVEL",
        "COLUMN_NAME","MAPPING_TEXT"
    ])

# ---------------- main ----------------
def main():
    now = datetime.datetime.now()
    ts  = now.strftime("%Y%m%d_%H%M%S")

    # set your input folder here
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
                 "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","TRANSFORMATION_LEVEL",
                 "COLUMN_NAME","MAPPING_TEXT"]
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
    main() ex
