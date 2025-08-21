# sapds_column_mapping_v3.py
# Column-level extractor for SAP DS XML export
# Captures source/transform/target columns with mapping text.

import re, os, glob, html, datetime
import pandas as pd
from lxml import etree as ET
from collections import defaultdict

# ---------------- utils ----------------
def strip_ns(tag): return re.sub(r"^\{.*\}", "", tag) if isinstance(tag,str) else ""
def lower(s): return (s or "").strip().lower()
def attrs_ci(e): return {k.lower():(v or "") for k,v in (getattr(e,"attrib",{}) or {}).items()}
def build_parent_map(root): return {c:p for p in root.iter() for c in p}
def ancestors(e, pm, lim=200):
    cur=e
    for _ in range(lim):
        if cur is None: break
        yield cur
        cur=pm.get(cur)
def norm(s): return re.sub(r"\s+"," ",(s or "").strip()).upper()

# ---------------- tag sets ----------------
DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")
CALLSTEP_TAGS = ("dicallstep","callstep")
SOURCE_CONTAINER_TAGS = ("didatabasetablesource","difilesource","diexcelsource","ditablesource")
TARGET_CONTAINER_TAGS = ("didatabasetabletarget","difiletarget","diexceltarget","ditabletarget")

# ---------------- context helpers ----------------
def _job_name_from_node(job_node):
    for ch in job_node.iter():
        if lower(strip_ns(getattr(ch,"tag","")))=="diattribute" and lower(ch.attrib.get("name",""))=="job_name":
            v=(ch.attrib.get("value") or "").strip()
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
    # simplified: one job per DF if possible
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
    for a in ancestors(e,pm):
        if lower(strip_ns(getattr(a,"tag",""))) in DF_TAGS: return True
    return False

def nearest_schema_name(e, pm):
    for a in ancestors(e,pm):
        if lower(strip_ns(getattr(a,"tag","")))=="dischema":
            nm=(attrs_ci(a).get("name") or attrs_ci(a).get("displayname") or "").strip()
            if nm: return nm
    return ""

# ---------------- global DITable catalog ----------------
def build_table_catalog(root):
    catalog=defaultdict(set)
    for n in root.iter():
        if lower(strip_ns(getattr(n,"tag","")))!="ditable": continue
        a=attrs_ci(n)
        tbl=(a.get("name") or a.get("table_name") or "").strip()
        owner=(a.get("owner") or a.get("ownername") or "").strip()
        ds=(a.get("datastore") or a.get("datastorename") or "").strip()
        if not (tbl and ds): continue
        key=(norm(ds),norm(owner),norm(tbl))
        for col in n.iter():
            if lower(strip_ns(getattr(col,"tag","")))=="dicolumn":
                cn=(attrs_ci(col).get("name") or "").strip()
                if cn: catalog[key].add(cn)
    return catalog

def extract_db_ref_from_schema(schema_node):
    for ch in schema_node.iter():
        t=lower(strip_ns(getattr(ch,"tag","")))
        if t in SOURCE_CONTAINER_TAGS or t in TARGET_CONTAINER_TAGS:
            a=attrs_ci(ch)
            ds=(a.get("datastorename") or a.get("datastore") or "").strip()
            owner=(a.get("ownername") or a.get("owner") or a.get("schema") or "").strip()
            tbl=(a.get("tablename") or a.get("table") or a.get("name") or "").strip()
            role="SOURCE" if t in SOURCE_CONTAINER_TAGS else "TARGET"
            if ds and tbl: return role,ds,owner,tbl
    return None

# ---------------- DF schema/columns ----------------
def compute_df_structures(root):
    df_schema_levels={}
    df_schema_cols=defaultdict(lambda:defaultdict(set))
    for df_node in root.iter():
        if lower(strip_ns(getattr(df_node,"tag",""))) not in DF_TAGS: continue
        df_name=(attrs_ci(df_node).get("name") or attrs_ci(df_node).get("displayname") or "").strip()
        if not df_name: continue
        lvlmap={};lvl=1
        for sc in df_node.iter():
            if lower(strip_ns(getattr(sc,"tag","")))!="dischema": continue
            sch=(attrs_ci(sc).get("name") or attrs_ci(sc).get("displayname") or "").strip()
            if not sch: continue
            if sch not in lvlmap: lvlmap[sch]=lvl;lvl+=1
            for el in sc.iter():
                if lower(strip_ns(getattr(el,"tag","")) )=="dielement":
                    cn=(attrs_ci(el).get("name") or "").strip()
                    if cn: df_schema_cols[df_name][sch].add(cn)
        df_schema_levels[df_name]=lvlmap
    return df_schema_levels,df_schema_cols

def transformation_type_by_tags(e, pm):
    for a in ancestors(e,pm):
        t=lower(strip_ns(getattr(a,"tag","")))
        if t in SOURCE_CONTAINER_TAGS: return "SOURCE"
        if t in TARGET_CONTAINER_TAGS: return "TARGET"
        if t=="dischema": break
    return "TRANSFORM"

# ---------------- parse one xml ----------------
def parse_single_xml(xml_path):
    parser=ET.XMLParser(huge_tree=True,recover=True)
    root=ET.parse(xml_path,parser=parser).getroot()
    pm=build_parent_map(root)

    job_to_project=build_job_to_project_map(root)
    df_to_project=build_df_project_map(root)
    df_to_job=build_df_job_map(root)

    table_catalog=build_table_catalog(root)
    df_schema_levels,df_schema_cols=compute_df_structures(root)

    rows=[]
    cur_proj=cur_job=cur_df=""

    # pass1: DIElements
    for e in root.iter():
        if not isinstance(e.tag,str): continue
        tag=lower(strip_ns(e.tag)); a=attrs_ci(e)
        if tag in PROJECT_TAGS: cur_proj=(a.get("name") or a.get("displayname") or cur_proj).strip()
        if tag in DF_TAGS: cur_df=(a.get("name") or a.get("displayname") or cur_df).strip()
        if tag in JOB_TAGS: cur_job=_job_name_from_node(e) or (a.get("name") or a.get("displayname") or cur_job).strip()
        if tag=="dielement" and in_dataflow(e,pm):
            col=(a.get("name") or "").strip()
            map_txt=""
            for ch in e.iter():
                if lower(strip_ns(getattr(ch,"tag","")) )=="diattribute" and lower(ch.attrib.get("name",""))=="ui_mapping_text":
                    map_txt=(ch.attrib.get("value") or "").strip(); break
            if map_txt: map_txt=html.unescape(map_txt).replace("\n"," ").strip()
            transform=nearest_schema_name(e,pm)
            t_type=transformation_type_by_tags(e,pm)
            lvl=df_schema_levels.get(cur_df,{}).get(transform,1)
            rows.append({
                "PROJECT_NAME":cur_proj,"JOB_NAME":cur_job,"DATAFLOW_NAME":cur_df,
                "TRANSFORMATION_NAME":transform,"TRANSFORMATION_TYPE":t_type,
                "TRANSFORMATION_LEVEL":lvl,"COLUMN_NAME":col,"MAPPING_TEXT":map_txt
            })
            if cur_df and transform and col: df_schema_cols[cur_df][transform].add(col)

    # pass2: add source/target schema columns
    for df_name,lmap in df_schema_levels.items():
        proj=df_to_project.get(df_name,""); job=df_to_job.get(df_name,"")
        for sc, lvl in lmap.items():
            dbref=None
            for df_node in root.iter():
                if lower(strip_ns(getattr(df_node,"tag",""))) not in DF_TAGS: continue
                if (attrs_ci(df_node).get("name") or "").strip()==df_name:
                    for s in df_node.iter():
                        if lower(strip_ns(getattr(s,"tag","")))=="dischema":
                            nm=(attrs_ci(s).get("name") or "").strip()
                            if nm==sc: dbref=extract_db_ref_from_schema(s)
            if not dbref: continue
            role,ds,owner,tbl=dbref
            key=(norm(ds),norm(owner),norm(tbl))
            cols=set(table_catalog.get(key,set())) or df_schema_cols[df_name].get(sc,set())
            for c in cols:
                rows.append({
                    "PROJECT_NAME":proj,"JOB_NAME":job,"DATAFLOW_NAME":df_name,
                    "TRANSFORMATION_NAME":sc,"TRANSFORMATION_TYPE":role,
                    "TRANSFORMATION_LEVEL":lvl,"COLUMN_NAME":c,"MAPPING_TEXT":""
                })

    if rows:
        df=pd.DataFrame(rows).drop_duplicates()
        return df
    return pd.DataFrame(columns=["PROJECT_NAME","JOB_NAME","DATAFLOW_NAME","TRANSFORMATION_NAME","TRANSFORMATION_TYPE","TRANSFORMATION_LEVEL","COLUMN_NAME","MAPPING_TEXT"])

# ---------------- main ----------------
def main():
    path=r"C:\Users\raksahu\Downloads\python\input\sap_ds_xml_files"
    all_files=glob.glob(os.path.join(path,"*.xml"))
    print(f"Found {len(all_files)} xml files")
    dfs=[parse_single_xml(f) for f in all_files]
    final=pd.concat(dfs,ignore_index=True) if dfs else pd.DataFrame()
    ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out=os.path.join(path,f"SAPDS_ALL_COLUMN_MAPPING_{ts}.csv")
    final.to_csv(out,index=False,encoding="utf-8-sig")
    print(f"Done. Wrote {out} | Rows: {len(final)}")

if __name__=="__main__":
    main()
