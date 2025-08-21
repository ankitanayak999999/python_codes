# sapds_column_mappings_v16.py
# Uses the SAME context logic as your table script to fill the first 4 columns
# Patched: adds TRANSFORMATION_TYPE and TRANSFORMATION_LEVEL

import re, os, glob, html, datetime
import pandas as pd
from lxml import etree as ET
from collections import defaultdict

# -------------------- small utils (same as your table script) --------------------
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

# ---------- tag sets (copied from your code) ----------
DF_TAGS       = ("didataflow","dataflow","dflow")
JOB_TAGS      = ("dijob","dibatchjob","job","batch_job")
PROJECT_TAGS  = ("diproject","project")
WF_TAGS       = ("diworkflow","workflow")
CALLSTEP_TAGS = ("dicallstep","callstep")

# --- NEW: for type detection ---
SOURCE_TAGS = ("didatabasetablesource","difilesource","diexcelsource")
TARGET_TAGS = ("didatabasetabletarget","difiletarget","diexceltarget")

# -------------------- context helpers (copied from your logic) --------------------
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

    # collect jobs / workflows
    for n in root.iter():
        t=lower(strip_ns(getattr(n,"tag","")))
        if t in JOB_TAGS:
            nm=_job_name_from_node(n)
            if nm: jobs[nm]=n
        elif t in WF_TAGS:
            nm=(n.attrib.get("name") or n.attrib.get("displayName") or "").strip()
            if nm: wfs[nm]=n

    # build edges via callsteps
    from collections import defaultdict as dd
    edges=dd(set)
    def add_edge(src_kind, src_name, dst_kind, dst_name):
        if src_name and dst_name:
            edges[(src_kind, canon(src_name))].add((dst_kind, canon(dst_name)))

    for cs in root.iter():
        if lower(strip_ns(getattr(cs,"tag",""))) not in CALLSTEP_TAGS: continue
        # climb to source job/wf
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

    # reachability to map df -> job
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

# in_dataflow (same behavior you rely on)
def in_dataflow(e, pm):
    for a in ancestors(e, pm, 200):
        if lower(strip_ns(getattr(a,"tag",""))) in DF_TAGS:
            return True
    return False

# nearest DISchema name for TRANSFORMATION_NAME
def nearest_schema_name(e, pm):
    for a in ancestors(e, pm, 200):
        if lower(strip_ns(getattr(a,"tag",""))) == "dischema":
            nm = (attrs_ci(a).get("name") or attrs_ci(a).get("displayname") or "").strip()
            if nm: return nm
    return ""

# --- NEW: detect type for a DIElement (SOURCE / TARGET / TRANSFORM) ---
def transformation_type_for(e, pm):
    for a in ancestors(e, pm, 200):
        t = lower(strip_ns(getattr(a,"tag","")))
        if t in SOURCE_TAGS: return "SOURCE"
        if t in TARGET_TAGS: return "TARGET"
        if t == "dischema":  return "TRANSFORM"
    return "UNKNOWN"

# --- NEW: build schema graph to compute levels per dataflow ---
def build_schema_graph(df_node):
    """
    Returns (schema_names, edges) where edges are (src_schema_name, dst_schema_name)
    Scans common link tags and tries id/name mapping.
    """
    # collect schema ids -> names
    id_to_name = {}
    names = set()
    for n in df_node.iter():
        if lower(strip_ns(getattr(n, "tag", ""))) == "dischema":
            a = attrs_ci(n)
            nm = (a.get("name") or a.get("displayname") or "").strip()
            nid = (a.get("id") or a.get("objid") or a.get("guid") or nm).strip()
            if nm:
                id_to_name[nid] = nm
                names.add(nm)

    def endpoint_to_name(val):
        if not val: return None
        v = val.strip()
        if v in id_to_name: return id_to_name[v]
        if v in names: return v
        last = v.split("/")[-1].split("\\")[-1].split(":")[-1]
        return last if last in names else None

    edges = []
    for n in df_node.iter():
        t = lower(strip_ns(getattr(n, "tag", "")))
        if t not in ("dilink","didataflowlink","link","connection","edge"):
            continue
        a = attrs_ci(n)
        for src_raw, dst_raw in (
            (a.get("from"), a.get("to")),
            (a.get("source"), a.get("target")),
            (a.get("start"), a.get("end")),
            (a.get("src"), a.get("dst")),
        ):
            if not src_raw or not dst_raw: 
                continue
            src = endpoint_to_name(src_raw)
            dst = endpoint_to_name(dst_raw)
            if src and dst and src != dst:
                edges.append((src, dst))
                break

    return names, edges

def levels_by_bfs(schema_names, edges):
    """Return dict: schema_name -> level (1-based)."""
    from collections import defaultdict, deque
    indeg = defaultdict(int)
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v); indeg[v] += 1
        if u not in indeg: indeg[u] = indeg[u]
    q = deque([n for n in schema_names if indeg.get(n, 0) == 0])
    level = {n: 1 for n in q}
    while q:
        u = q.popleft()
        for v in adj[u]:
            cand = level[u] + 1
            if level.get(v, 0) < cand: level[v] = cand
            indeg[v] -= 1
            if indeg[v] == 0: q.append(v)
    for n in schema_names: level.setdefault(n, 1)
    return level

# -------------------- column-mapping parser --------------------
def parse_single_xml(xml_path: str) -> pd.DataFrame:
    parser = ET.XMLParser(huge_tree=True, recover=True)
    tree   = ET.parse(xml_path, parser=parser)
    root   = tree.getroot()
    pm     = build_parent_map(root)

    # reuse the SAME context resolution you already use
    job_to_project = build_job_to_project_map(root)
    df_to_project  = build_df_project_map(root)
    df_to_job      = build_df_job_map(root)

    # NEW: pre-compute levels per dataflow (graph-based)
    df_levels_cache = {}
    for df_node in root.iter():
        if lower(strip_ns(getattr(df_node, "tag", ""))) in DF_TAGS:
            a = attrs_ci(df_node)
            df_name = (a.get("name") or a.get("displayname") or "").strip()
            if not df_name: 
                continue
            schema_names, edges = build_schema_graph(df_node)
            if schema_names:
                df_levels_cache[df_name] = levels_by_bfs(schema_names, edges)

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

    # walk once to keep current DF/job/project (matches your approach)
    for e in root.iter():
        if not isinstance(e.tag, str): continue
        tag = lower(strip_ns(e.tag))
        a   = attrs_ci(e)

        if tag in PROJECT_TAGS: cur_proj=(a.get("name") or a.get("displayname") or cur_proj).strip()
        if tag in DF_TAGS:      cur_df  =(a.get("name") or a.get("displayname") or cur_df).strip()
        if tag in JOB_TAGS:     cur_job = _job_name_from_node(e) or (a.get("name") or a.get("displayname") or cur_job).strip()

        # Only collect column rows inside a dataflow
        if tag == "dielement" and in_dataflow(e, pm):
            proj, job, df = context_for(e)
            col_name = (a.get("name") or "").strip()

            # mapping text under DIAttributes (ui_mapping_text)
            map_txt = ""
            for ch in e.iter():
                if lower(strip_ns(getattr(ch,"tag",""))) == "diattribute" and lower(getattr(ch, "attrib", {}).get("name","")) == "ui_mapping_text":
                    map_txt = (getattr(ch, "attrib", {}).get("value") or "").strip()
                    if map_txt: break
            if map_txt:
                map_txt = html.unescape(map_txt).replace("\r", " ").replace("\n", " ").strip()

            transform = nearest_schema_name(e, pm)
            t_type    = transformation_type_for(e, pm)

            # figure level: prefer graph; fallback heuristic
            lvl = None
            if df in df_levels_cache and transform:
                lvl = df_levels_cache[df].get(transform)
            if not lvl:
                if t_type == "SOURCE":
                    lvl = 1
                elif t_type == "TARGET":
                    if df in df_levels_cache and df_levels_cache[df]:
                        lvl = max(df_levels_cache[df].values()) + 1
                    else:
                        lvl = 3
                else:
                    lvl = 2

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

    return pd.DataFrame(rows, columns=[
        "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
        "TRANSFORMATION_NAME","TRANSFORMATION_TYPE","TRANSFORMATION_LEVEL",
        "COLUMN_NAME","MAPPING_TEXT"
    ])

# -------------------- main (v3 style like your script) --------------------
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
        columns=["PROJECT_NAME","JOB_NAME","DATAFLOW_NAME","TRANSFORMATION_NAME","TRANSFORMATION_TYPE","TRANSFORMATION_LEVEL","COLUMN_NAME","MAPPING_TEXT"]
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
