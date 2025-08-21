# sapds_column_mappings_v3.py
# Column-level extractor aligned with your table-level logic
# - TRANSFORMATION_TYPE: SOURCE / TRANSFORM / TARGET
# - TRANSFORMATION_LEVEL: 1..N by schema order within each Dataflow
# - Columns for Source/Target pulled from global <DITable> catalog
# - Targets also back-filled from DIInputView if table catalog is missing

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
def norm(s): return re.sub(r"\s+", " ", (s or "").strip()).upper()

# ---------- tag sets ----------
DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")
CALLSTEP_TAGS = ("dicallstep","callstep")

# exact source/target container tags (same family you used on table script)
SOURCE_CONTAINER_TAGS = ("didatabasetablesource","difilesource","diexcelsource")
TARGET_CONTAINER_TAGS = ("didatabasetabletarget","difiletarget","diexceltarget")

# -------------------- context helpers (same as your table script) --------------------
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

# -------------------- NEW: global DITable catalog --------------------
def build_table_catalog(root):
    """
    Build { (DS, OWNER, TABLE) -> set(columns) } from global <DITable> blocks.
    Supports attributes: name / table_name, owner / ownerName, datastore / datastoreName.
    """
    catalog = defaultdict(set)
    for n in root.iter():
        if lower(strip_ns(getattr(n, "tag", ""))) != "ditable":
            continue
        a = attrs_ci(n)
        tbl  = (a.get("name") or a.get("table_name") or "").strip()
        owner= (a.get("owner") or a.get("ownername") or "").strip()
        ds   = (a.get("datastore") or a.get("datastorename") or "").strip()
        if not (tbl and ds):
            continue
        key = (norm(ds), norm(owner), norm(tbl))
        for col in n.iter():
            if lower(strip_ns(getattr(col, "tag", ""))) == "dicolumn":
                cn = (attrs_ci(col).get("name") or "").strip()
                if cn:
                    catalog[key].add(cn)
    return catalog

def extract_db_ref_from_schema(schema_node):
    """
    If schema contains a DB/file/Excel source/target child, return (role, ds, owner, table).
    role: 'SOURCE' or 'TARGET'
    """
    for ch in schema_node.iter():
        t = lower(strip_ns(getattr(ch, "tag", "")))
        if t in SOURCE_CONTAINER_TAGS or t in TARGET_CONTAINER_TAGS:
            a = attrs_ci(ch)
            ds = (a.get("datastorename") or a.get("datastore") or "").strip()
            owner = (a.get("ownername") or a.get("owner") or "").strip()
            tbl = (a.get("tablename") or a.get("table") or a.get("name") or "").strip()
            role = "SOURCE" if t in SOURCE_CONTAINER_TAGS else "TARGET"
            if ds and tbl:
                return role, ds, owner, tbl
    return None

# ---- simple per-DF schema order + DIInputView mapping ----
def compute_df_structures(root):
    """
    Returns:
      - df_schema_levels: {df: {schema: level}}
      - df_schema_cols:   {df: {schema: set(cols)}} (explicit DIElements)
      - df_target_inputs: {df: {target_schema: set(input_schema_names)}} via DIInputView
    """
    df_schema_levels = {}
    df_schema_cols   = defaultdict(lambda: defaultdict(set))
    df_target_inputs = defaultdict(lambda: defaultdict(set))

    for df_node in root.iter():
        if lower(strip_ns(getattr(df_node, "tag", ""))) not in DF_TAGS:
            continue
        a = attrs_ci(df_node)
        df_name = (a.get("name") or a.get("displayname") or "").strip()
        if not df_name: continue

        level_map, next_level = {}, 1
        for sc in df_node.iter():
            if lower(strip_ns(getattr(sc, "tag", ""))) != "dischema":
                continue
            # order -> level
            sch_name = (attrs_ci(sc).get("name") or attrs_ci(sc).get("displayname") or "").strip()
            if not sch_name: 
                continue
            if sch_name not in level_map:
                level_map[sch_name] = next_level
                next_level += 1

            # collect explicit DIElements under this schema
            for el in sc.iter():
                if lower(strip_ns(getattr(el, "tag", ""))) == "dielement":
                    cn = (attrs_ci(el).get("name") or "").strip()
                    if cn:
                        df_schema_cols[df_name][sch_name].add(cn)

            # record DIInputView(s) for targets
            for iv in sc.iter():
                if lower(strip_ns(getattr(iv, "tag", ""))) in ("diinputview","inputview"):
                    inp = (attrs_ci(iv).get("name") or "").strip()
                    if inp:
                        df_target_inputs[df_name][sch_name].add(inp)

        if level_map:
            df_schema_levels[df_name] = level_map

    return df_schema_levels, df_schema_cols, df_target_inputs

# ---- classification helpers ----
def transformation_type_by_tags(e, pm):
    # look for source/target container ancestors
    for a in ancestors(e, pm, 200):
        t = lower(strip_ns(getattr(a, "tag", "")))
        if t in SOURCE_CONTAINER_TAGS: return "SOURCE"
        if t in TARGET_CONTAINER_TAGS: return "TARGET"
        if t == "dischema": break
    # generic fallbacks via tag/attr text
    for a in ancestors(e, pm, 200):
        t = lower(strip_ns(getattr(a, "tag", "")))
        if "source" in t: return "SOURCE"
        if "target" in t: return "TARGET"
        at = attrs_ci(a)
        for k in ("role","type","schema_type","objecttype","transformtype","category"):
            v = lower(at.get(k))
            if v:
                if "source" in v: return "SOURCE"
                if "target" in v: return "TARGET"
    return "TRANSFORM"

# -------------------- main column parser --------------------
def parse_single_xml(xml_path: str) -> pd.DataFrame:
    parser = ET.XMLParser(huge_tree=True, recover=True)
    root   = ET.parse(xml_path, parser=parser).getroot()
    pm     = build_parent_map(root)

    # Context maps
    job_to_project = build_job_to_project_map(root)
    df_to_project  = build_df_project_map(root)
    df_to_job      = build_df_job_map(root)

    # Structures
    table_catalog = build_table_catalog(root)                       # global tables -> columns
    df_schema_levels, df_schema_cols, df_target_inputs = compute_df_structures(root)

    # df -> (proj, job) context
    df_context = {}
    for df_node in root.iter():
        if lower(strip_ns(getattr(df_node, "tag", ""))) in DF_TAGS:
            a = attrs_ci(df_node)
            df_name = (a.get("name") or a.get("displayname") or "").strip()
            if not df_name: continue
            job = df_to_job.get(df_name, "") or ""
            proj = job_to_project.get(job, "") or df_to_project.get(df_name, "")
            df_context[df_name] = (proj, job)

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
        if not job:
            job = df_to_job.get(df, "") or cur_job
        if not proj:
            proj = job_to_project.get(job, "") or df_to_project.get(df, "") or cur_proj
        return proj or "", job or "", df or ""

    rows = []

    # Pass 1: explicit DIElement rows
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

            map_txt = ""
            for ch in e.iter():
                if lower(strip_ns(getattr(ch,"tag",""))) == "diattribute" and \
                   lower(getattr(ch, "attrib", {}).get("name","")) == "ui_mapping_text":
                    map_txt = (getattr(ch, "attrib", {}).get("value") or "").strip()
                    if map_txt: break
            if map_txt:
                map_txt = html.unescape(map_txt).replace("\r", " ").replace("\n", " ").strip()

            transform = nearest_schema_name(e, pm)
            t_type    = transformation_type_by_tags(e, pm)

            lvl = 1
            if df and transform and df in df_schema_levels:
                lvl = df_schema_levels[df].get(transform, 1)

            rows.append({
                "PROJECT_NAME": proj,
                "JOB_NAME": job,
                "DATAFLOW_NAME": df,
                "TRANSFORMATION_NAME": transform,
                "TRANSFORMATION_TYPE": t_type,
                "TRANSFORMATION_LEVEL": int(lvl),
                "COLUMN_NAME": col_name,
                "MAPPING_TEXT": map_txt
            })
            if df and transform and col_name:
                df_schema_cols[df][transform].add(col_name)

    # Pass 2: add columns for Source/Target schemas by reading table catalog
    # We walk each DF's DISchema; if it references a table (via child source/target),
    # we fetch its column set and emit rows (blank mapping text).
    for df_node in root.iter():
        if lower(strip_ns(getattr(df_node, "tag", ""))) not in DF_TAGS:
            continue
        df_name = (attrs_ci(df_node).get("name") or attrs_ci(df_node).get("displayname") or "").strip()
        if not df_name: 
            continue
        proj, job = df_context.get(df_name, ("",""))
        for sc in df_node.iter():
            if lower(strip_ns(getattr(sc, "tag", ""))) != "dischema":
                continue
            sch_name = (attrs_ci(sc).get("name") or attrs_ci(sc).get("displayname") or "").strip()
            if not sch_name:
                continue

            dbref = extract_db_ref_from_schema(sc)
            if not dbref:
                continue
            role, ds, owner, tbl = dbref
            key = (norm(ds), norm(owner), norm(tbl))
            cols = table_catalog.get(key, set())

            # If catalog missing, try DIInputView back-fill for TARGET
            if not cols and role == "TARGET":
                for iv in sc.iter():
                    if lower(strip_ns(getattr(iv, "tag", ""))) in ("diinputview","inputview"):
                        inp = (attrs_ci(iv).get("name") or "").strip()
                        if inp:
                            cols |= df_schema_cols[df_name].get(inp, set())

            if not cols:
                continue  # nothing to add

            lvl = 1
            if df_name in df_schema_levels:
                lvl = df_schema_levels[df_name].get(sch_name, 1)

            for col in sorted(cols):
                rows.append({
                    "PROJECT_NAME": proj,
                    "JOB_NAME": job,
                    "DATAFLOW_NAME": df_name,
                    "TRANSFORMATION_NAME": sch_name,
                    "TRANSFORMATION_TYPE": role,             # SOURCE or TARGET
                    "TRANSFORMATION_LEVEL": int(lvl),
                    "COLUMN_NAME": col,
                    "MAPPING_TEXT": ""                        # per your request
                })

    # Deduplicate identical rows (same DF / schema / column / type)
    if rows:
        df = pd.DataFrame(rows)
        df["__k__"] = (
            df["PROJECT_NAME"].astype(str) + "||" +
            df["JOB_NAME"].astype(str) + "||" +
            df["DATAFLOW_NAME"].astype(str) + "||" +
            df["TRANSFORMATION_NAME"].astype(str) + "||" +
            df["TRANSFORMATION_TYPE"].astype(str) + "||" +
            df["COLUMN_NAME"].astype(str)
        )
        df = df.drop_duplicates(subset="__k__").drop(columns="__k__")
        # Final column order
        return df[[
            "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
            "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","TRANSFORMATION_LEVEL",
            "COLUMN_NAME","MAPPING_TEXT"
        ]].reset_index(drop=True)
    else:
        return pd.DataFrame(columns=[
            "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
            "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","TRANSFORMATION_LEVEL",
            "COLUMN_NAME","MAPPING_TEXT"
        ])

# -------------------- main --------------------
def main():
    now = datetime.datetime.now()
    ts  = now.strftime("%Y%m%d_%H%M%S")

    # keep your paths here
    path = r"C:\Users\raksahu\Downloads\python\input\sap_ds_xml_files"
    single_file = fr"{path}\export_af.xml"   # optional

    all_files = glob.glob(os.path.join(path, "*.xml"))
    if single_file in all_files:
        all_files = [single_file]

    print(f"total number of files present in the path ({len(all_files)})")
    print(all_files)

    df_list=[]
    for i, file in enumerate(all_files):
        print(f"Row Number:{i}--{file}")
        df_list.append(parse_single_xml(file))

    final_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame(
        columns=[
            "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
            "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","TRANSFORMATION_LEVEL",
            "COLUMN_NAME","MAPPING_TEXT"
        ]
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
