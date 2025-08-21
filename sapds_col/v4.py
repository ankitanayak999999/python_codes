# sapds_column_mapping_v3_debug.py
# Column-level extractor for SAP DS XML exports (STRICT 3-key match)
# Adds detailed debug logs to show why Source/Target joins miss.

import re, os, glob, html, datetime
import pandas as pd
from lxml import etree as ET
from collections import defaultdict

DEBUG = True  # leave True for one run to collect debug CSVs

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
def canon(s: str) -> str:
    s = (s or "").strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s.upper()

# ---------------- tag sets ----------------
DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")
CALLSTEP_TAGS = ("dicallstep","callstep")

SOURCE_CONTAINER_TAGS = (
    "didatabasetablesource", "difilesource", "diexcelsource",
    "ditablesource"
)
TARGET_CONTAINER_TAGS = (
    "didatabasetabletarget", "difiletarget", "diexceltarget",
    "ditabletarget"
)

# ---------------- context helpers (aligned with table-level code) ----------------
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
            if lower(strip_ns(getattr(ref,"tag","")))=="dijobref":
                jn=(ref.attrib.get("name") or ref.attrib.get("displayName") or "").strip()
                if jn: j2p.setdefault(jn, proj)
    return j2p

def build_df_project_map(root):
    df_names=collect_df_names(root)
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
                if dn: df_proj.setdefault(dn,pnm)
    if len(projects)==1:
        only=projects[0][0]
        for dn in df_names: df_proj.setdefault(dn, only)
    return df_proj

def build_df_job_map(root):
    jobs={}
    for n in root.iter():
        if lower(strip_ns(getattr(n,"tag",""))) in JOB_TAGS:
            nm=_job_name_from_node(n)
            if nm: jobs[nm]=n
    df_names=collect_df_names(root)
    df_job={}
    if len(jobs)==1:
        only=list(jobs.keys())[0]
        for d in df_names: df_job[d]=only
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

# ---------------- global <DITable> catalog (STRICT 3-key) ----------------
def build_table_catalog(root):
    """
    Returns:
      by3[(DS, OWNER, TABLE)] -> set(columns)
      also returns a flat list of catalog rows for debug CSV
    Keys are canonicalized (uppercased, trimmed).
    """
    by3 = defaultdict(set)
    debug_rows = []

    for n in root.iter():
        if lower(strip_ns(getattr(n, "tag", ""))) != "ditable":
            continue
        a = attrs_ci(n)

        tbl_raw   = a.get("name") or a.get("table_name") or ""
        owner_raw = a.get("owner") or a.get("ownername") or ""
        ds_raw    = a.get("datastore") or a.get("datastorename") or ""

        tblC, ownerC, dsC = canon(tbl_raw), canon(owner_raw), canon(ds_raw)
        if not (tblC and ownerC and dsC):
            continue

        cols = set()
        for c in n.iter():
            if lower(strip_ns(getattr(c, "tag", ""))) == "dicolumn":
                cn = (attrs_ci(c).get("name") or "").strip()
                if cn:
                    cols.add(cn)

        if cols:
            by3[(dsC, ownerC, tblC)] |= cols
            debug_rows.append({
                "CATALOG_DS_CANON": dsC,
                "CATALOG_OWNER_CANON": ownerC,
                "CATALOG_TABLE_CANON": tblC,
                "CATALOG_DS_RAW": ds_raw,
                "CATALOG_OWNER_RAW": owner_raw,
                "CATALOG_TABLE_RAW": tbl_raw,
                "CATALOG_COL_COUNT": len(cols)
            })

    return by3, debug_rows

def extract_db_ref_from_schema(schema_node):
    """
    Return (role, ds_raw, owner_raw, table_raw, dsC, ownerC, tableC)
    for a Source/Target container beneath the schema.
    """
    for ch in schema_node.iter():
        t = lower(strip_ns(getattr(ch, "tag", "")))
        if t in SOURCE_CONTAINER_TAGS or t in TARGET_CONTAINER_TAGS:
            a = attrs_ci(ch)
            ds    = (a.get("datastorename") or a.get("datastore") or "").strip()
            owner = (a.get("ownername")     or a.get("owner")     or a.get("schema") or "").strip()
            tbl   = (a.get("tablename")     or a.get("table")     or a.get("name")   or "").strip()
            role  = "SOURCE" if t in SOURCE_CONTAINER_TAGS else "TARGET"
            if tbl:
                return role, ds, owner, tbl, canon(ds), canon(owner), canon(tbl)
    return None

# ---------------- DF structures: schema order + explicit DIElements + db refs ----------------
def compute_df_structures(root):
    """
    Returns:
      df_levels: {df: {schema: level}}
      df_cols:   {df: {schema: set(cols)}}  # DIElements seen in that schema
      df_dbref:  {df: {schema: (role, ds_raw, owner_raw, tbl_raw, dsC, ownerC, tblC)}}
    """
    df_levels = {}
    df_cols   = defaultdict(lambda: defaultdict(set))
    df_dbref  = defaultdict(dict)

    for df_node in root.iter():
        if lower(strip_ns(getattr(df_node, "tag", ""))) not in DF_TAGS:
            continue
        df_name = (attrs_ci(df_node).get("name") or attrs_ci(df_node).get("displayname") or "").strip()
        if not df_name: continue

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

            # strict Source/Target reference under this schema
            ref = extract_db_ref_from_schema(sc)
            if ref:
                df_dbref[df_name][sch] = ref

        if levels:
            df_levels[df_name] = levels

    return df_levels, df_cols, df_dbref

def transformation_type_by_tags(e, pm):
    for a in ancestors(e, pm, 200):
        t = lower(strip_ns(getattr(a, "tag", "")))
        if t in SOURCE_CONTAINER_TAGS: return "SOURCE"
        if t in TARGET_CONTAINER_TAGS: return "TARGET"
        if t == "dischema": break
    return "TRANSFORM"

# ---------------- parse one xml ----------------
def parse_single_xml(xml_path: str):
    parser = ET.XMLParser(huge_tree=True, recover=True)
    root   = ET.parse(xml_path, parser=parser).getroot()
    pm     = build_parent_map(root)

    job_to_project = build_job_to_project_map(root)
    df_to_project  = build_df_project_map(root)
    df_to_job      = build_df_job_map(root)

    table_by3, catalog_debug_rows = build_table_catalog(root)
    df_levels, df_cols, df_dbref  = compute_df_structures(root)

    # df -> (proj, job)
    df_context = {}
    for df in df_levels.keys():
        job = df_to_job.get(df, "") or ""
        proj = job_to_project.get(job, "") or df_to_project.get(df, "")
        df_context[df] = (proj, job)

    rows = []
    debug_unmatched = []   # list of dicts

    # Pass 1: DIElements (TRANSFORM / S/T if DIElements exist there)
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

    # Pass 2: strict 3-key SOURCE/TARGET from catalog
    for df_name, schemas in df_levels.items():
        proj, job = df_context.get(df_name, ("",""))
        for sch_name, lvl in schemas.items():
            ref = df_dbref.get(df_name, {}).get(sch_name)
            if not ref:
                continue

            role, ds_raw, owner_raw, tbl_raw, dsC, ownerC, tblC = ref
            key = (dsC, ownerC, tblC)
            cols = table_by3.get(key, set())

            if not cols and DEBUG:
                # collect helpful hints
                # scan catalog rows to find candidates
                same_owner_tbl = []
                same_tbl = []
                for (cds, cowner, ctbl), ccols in table_by3.items():
                    if cowner == ownerC and ctbl == tblC and cds != dsC:
                        same_owner_tbl.append((cds, cowner, ctbl, len(ccols)))
                    if ctbl == tblC and not (cowner == ownerC and cds == dsC):
                        same_tbl.append((cds, cowner, ctbl, len(ccols)))

                debug_unmatched.append({
                    "DF": df_name,
                    "SCHEMA": sch_name,
                    "ROLE": role,
                    "DS_RAW": ds_raw, "OWNER_RAW": owner_raw, "TABLE_RAW": tbl_raw,
                    "DS_CANON": dsC, "OWNER_CANON": ownerC, "TABLE_CANON": tblC,
                    "FOUND_EXACT": False,
                    "NEAR_MATCH_OWNER+TABLE_COUNT": len(same_owner_tbl),
                    "NEAR_MATCH_TABLE_COUNT": len(same_tbl),
                    "NEAR_MATCH_OWNER+TABLE_SAMPLES": "; ".join([f"{d}/{o}/{t}:{c}" for d,o,t,c in same_owner_tbl[:5]]),
                    "NEAR_MATCH_TABLE_SAMPLES": "; ".join([f"{d}/{o}/{t}:{c}" for d,o,t,c in same_tbl[:5]])
                })

            if not cols:
                continue  # strict mode: skip if not found

            for col_name in sorted(cols):
                rows.append({
                    "PROJECT_NAME": proj,
                    "JOB_NAME": job,
                    "DATAFLOW_NAME": df_name,
                    "TRANSFORMATION_NAME": sch_name,
                    "TRANSFORMATION_TYPE": role,
                    "TRANSFORMATION_LEVEL": int(lvl),
                    "COLUMN_NAME": col_name,
                    "MAPPING_TEXT": ""
                })

    # return DF + optional debug info
    out_df = (pd.DataFrame(rows)
              if rows else pd.DataFrame(columns=[
                    "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
                    "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","TRANSFORMATION_LEVEL",
                    "COLUMN_NAME","MAPPING_TEXT"
                ]))
    debug_df_catalog = pd.DataFrame(catalog_debug_rows)
    debug_df_unmatched = pd.DataFrame(debug_unmatched)
    return out_df, debug_df_catalog, debug_df_unmatched

# ---------------- main ----------------
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
    cat_frames=[]
    unmatched_frames=[]
    for i, file in enumerate(all_files):
        print(f"Row Number:{i}--{file}")
        df, df_cat, df_unm = parse_single_xml(file)
        out_frames.append(df)
        if DEBUG:
            df_cat["XML_FILE"] = os.path.basename(file)
            df_unm["XML_FILE"] = os.path.basename(file)
            cat_frames.append(df_cat)
            unmatched_frames.append(df_unm)

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

    if DEBUG:
        # Write debug aids
        cat_df = pd.concat(cat_frames, ignore_index=True) if cat_frames else pd.DataFrame(
            columns=["CATALOG_DS_CANON","CATALOG_OWNER_CANON","CATALOG_TABLE_CANON",
                     "CATALOG_DS_RAW","CATALOG_OWNER_RAW","CATALOG_TABLE_RAW","CATALOG_COL_COUNT","XML_FILE"]
        )
        unm_df = pd.concat(unmatched_frames, ignore_index=True) if unmatched_frames else pd.DataFrame(
            columns=["DF","SCHEMA","ROLE","DS_RAW","OWNER_RAW","TABLE_RAW",
                     "DS_CANON","OWNER_CANON","TABLE_CANON",
                     "FOUND_EXACT","NEAR_MATCH_OWNER+TABLE_COUNT","NEAR_MATCH_TABLE_COUNT",
                     "NEAR_MATCH_OWNER+TABLE_SAMPLES","NEAR_MATCH_TABLE_SAMPLES","XML_FILE"]
        )
        cat_path = fr"{path}\SAPDS_TABLE_CATALOG_KEYS_{ts}.csv"
        unm_path = fr"{path}\SAPDS_COLUMN_DEBUG_UNMATCHED_{ts}.csv"
        cat_df.to_csv(cat_path, index=False, encoding="utf-8-sig")
        unm_df.to_csv(unm_path, index=False, encoding="utf-8-sig")
        print(f"[DEBUG] Catalog keys: {cat_path} | rows={len(cat_df)}")
        print(f"[DEBUG] Unmatched refs: {unm_path} | rows={len(unm_df)}")

if __name__ == "__main__":
    main()
