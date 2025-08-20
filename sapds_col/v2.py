# xml_parser_v16_columns.py
import os, glob, html, datetime
import pandas as pd
import xml.etree.ElementTree as ET

# ---------- core parser (single file) ----------
def parse_single_xml(xml_path: str) -> pd.DataFrame:
    """
    Parse one SAP DS export XML and return a DataFrame with:
    PROJECT_NAME, JOB_NAME, DATAFLOW_NAME, TRANSFORMATION_NAME, COLUMN_NAME, MAPPING_TEXT
    Only reads tags under <DITSchema>/<DIElement>/<DIAttribute name="ui_mapping_text">.
    Robust for large files using iterparse; frees memory as it goes.
    """
    rows = []

    # If you want to infer project/job/df from filename, do it here
    project_name, job_name, dataflow_name = "", "", ""
    # e.g., uncomment to use folder/file patterns:
    # base = os.path.basename(xml_path)
    # project_name = os.path.basename(os.path.dirname(xml_path))
    # job_name = ""
    # dataflow_name = os.path.splitext(base)[0]

    # Stream parse
    # We track the current <DITSchema name="..."> as TRANSFORMATION_NAME
    current_transform = ""

    for event, elem in ET.iterparse(xml_path, events=("start", "end")):
        tag = elem.tag

        # Entering a schema -> capture its name
        if event == "start" and tag == "DITSchema":
            current_transform = elem.attrib.get("name", "") or ""

        # On DIElement end, harvest the column + mapping_text from its descendants
        if event == "end" and tag == "DIElement":
            column_name = elem.attrib.get("name", "") or ""
            mapping_text = ""

            # Look for DIAttribute name="ui_mapping_text"
            for attr in elem.findall(".//DIAttribute"):
                if attr.attrib.get("name") == "ui_mapping_text":
                    mapping_text = attr.attrib.get("value", "") or ""
                    break

            # Clean up mapping text: unescape &quot; etc. and squash newlines
            if mapping_text:
                mapping_text = html.unescape(mapping_text).replace("\r", " ").replace("\n", " ").strip()

            # Append row
            rows.append({
                "PROJECT_NAME": project_name,
                "JOB_NAME": job_name,
                "DATAFLOW_NAME": dataflow_name,
                "TRANSFORMATION_NAME": current_transform,
                "COLUMN_NAME": column_name,
                "MAPPING_TEXT": mapping_text
            })

            # Free memory
            elem.clear()

        # Leaving a schema -> reset name
        if event == "end" and tag == "DITSchema":
            current_transform = ""
            elem.clear()

    return pd.DataFrame(rows, columns=[
        "PROJECT_NAME","JOB_NAME","DATAFLOW_NAME",
        "TRANSFORMATION_NAME","COLUMN_NAME","MAPPING_TEXT"
    ])

# -------------------- main --------------------
def main():
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")

    # ---- keep your paths here (same style as your v16) ----
    path = r"C:\Users\raksahu\Downloads\python\input\sap_ds_xml_files"   # <--- folder with XMLs
    single_file = fr"{path}\export_af.xml"                               # <--- optional single file

    all_files = glob.glob(os.path.join(path, "*.xml"))
    if single_file and os.path.isfile(single_file):
        # If a specific file exists, only parse that (v16 behavior you showed)
        all_files = [single_file]

    print(f"Total XML files to parse: {len(all_files)}")
    for i, f in enumerate(all_files, 1):
        print(f"[{i}/{len(all_files)}] {f}")

    df_list = []
    for file in all_files:
        df = parse_single_xml(file)
        df_list.append(df)

    final_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame(
        columns=["PROJECT_NAME","JOB_NAME","DATAFLOW_NAME","TRANSFORMATION_NAME","COLUMN_NAME","MAPPING_TEXT"]
    )

    # ---- save outputs (CSV + Excel) ----
    out_base = os.path.join(path, f"SAPDS_ALL_COLUMN_MAPPING_{timestamp}")
    csv_path = out_base + ".csv"
    xlsx_path = out_base + ".xlsx"

    # CSV with quoting-safe newlines handled above; add utf-8-sig for Excel
    final_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    # Also Excel (some users prefer)
    try:
        final_df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"Excel write skipped: {e}")

    print(f"\nRows extracted: {len(final_df)}")
    print(f"CSV:   {csv_path}")
    print(f"Excel: {xlsx_path}")

if __name__ == "__main__":
    main()
