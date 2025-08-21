import os
import glob
import datetime
import pandas as pd
import xml.etree.ElementTree as ET


# -------------------------------
# Helper: Extract DI table catalog (all DI tables & columns)
# -------------------------------
def extract_ditable_catalog(root):
    catalog = {}
    for table in root.findall(".//DITable"):
        key = (
            table.attrib.get("datastore", ""),
            table.attrib.get("owner", ""),
            table.attrib.get("name", "")
        )
        cols = []
        for col in table.findall(".//DIColumn"):
            cols.append(col.attrib.get("name", ""))
        catalog[key] = set(cols)
    return catalog


# -------------------------------
# Helper: Collect source/target refs (Database, File, Excel)
# -------------------------------
def collect_source_target_refs(root):
    refs = []
    # Database Sources
    for node in root.findall(".//DIDatabaseTableSource"):
        refs.append({
            "kind": "DATABASE",
            "df": node.attrib.get("outputView", ""),
            "schema": node.attrib.get("tableName", ""),
            "owner": node.attrib.get("ownerName", ""),
            "dsc": node.attrib.get("datastoreName", ""),
            "role": "SOURCE"
        })
    # Database Targets
    for node in root.findall(".//DITable"):
        refs.append({
            "kind": "DATABASE",
            "df": node.attrib.get("name", ""),
            "schema": node.attrib.get("name", ""),
            "owner": node.attrib.get("owner", ""),
            "dsc": node.attrib.get("datastore", ""),
            "role": "TARGET"
        })
    # Files / Excel
    for node in root.findall(".//DIFileSource") + root.findall(".//DIExcelSource"):
        refs.append({
            "kind": "FILE",
            "df": node.attrib.get("outputView", ""),
            "schema": node.attrib.get("name", ""),
            "owner": "",
            "dsc": "",
            "role": "SOURCE"
        })
    for node in root.findall(".//DIFileTarget") + root.findall(".//DIExcelTarget"):
        refs.append({
            "kind": "FILE",
            "df": node.attrib.get("name", ""),
            "schema": node.attrib.get("name", ""),
            "owner": "",
            "dsc": "",
            "role": "TARGET"
        })
    return refs


# -------------------------------
# Parser for single XML
# -------------------------------
def parse_single_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()

    rows = []

    # Collect DI tables and references
    table_catalog = extract_ditable_catalog(root)
    st_refs = collect_source_target_refs(root)

    # Expand
    for r in st_refs:
        proj = ""
        job = ""
        if r["kind"] == "DATABASE":
            key = (r["dsc"], r["owner"], r["schema"])
            cols = table_catalog.get(key, set())
            for col in sorted(cols):
                row = {
                    "PROJECT_NAME": proj,
                    "JOB_NAME": job,
                    "DATAFLOW_NAME": r["df"],
                    "TRANSFORMATION_NAME": r["schema"],  # schema node name
                    "TRANSFORMATION_TYPE": r["role"],   # SOURCE or TARGET
                    "COLUMN_NAME": col,
                    "MAPPING_TEXT": ""
                }
                rows.append(row)  # <-- Append database rows
        else:
            # File / Excel: emit row without column expansion
            row = {
                "PROJECT_NAME": proj,
                "JOB_NAME": job,
                "DATAFLOW_NAME": r["df"],
                "TRANSFORMATION_NAME": r["schema"],
                "TRANSFORMATION_TYPE": r["role"],
                "COLUMN_NAME": "",
                "MAPPING_TEXT": ""
            }
            rows.append(row)  # <-- Append file/excel rows

    return pd.DataFrame(rows)


# -------------------------------
# MAIN (exactly as in your screenshot)
# -------------------------------
def main():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    path = r"C:\Users\raksahu\Downloads\python\input\sap_ds_xml_files"
    single_file = fr"{path}\export_af.xml"  # optional pin

    all_files = glob.glob(os.path.join(path, "*.xml"))
    if single_file in all_files:
        all_files = [single_file]

    print(f"total number of files present in the path ({len(all_files)})")
    print(all_files)

    out_frames = []
    for i, file in enumerate(all_files):
        print(f"Row Number:{i}--{file}")
        out_frames.append(parse_single_xml(file))

    final_df = pd.concat(out_frames, ignore_index=True)

    rename_mapping = {
        'PROJECT_NAME': 'PROJECT_NAME',
        'JOB_NAME': 'JOB_NAME',
        'DATAFLOW_NAME': 'DATAFLOW_NAME',
        'TRANSFORMATION_NAME': 'TRANSFORMATION_NAME',
        'TRANSFORMATION_TYPE': 'TRANSFORMATION_TYPE',
        'COLUMN_NAME': 'COLUMN_NAME',
        'MAPPING_TEXT': 'MAPPING_TEXT'
    }

    final_df = final_df.rename(columns=rename_mapping)[list(rename_mapping.values())]
    key_cols = ["PROJECT_NAME","JOB_NAME","DATAFLOW_NAME","TRANSFORMATION_NAME","COLUMN_NAME"]
    final_df["RECORD_KEY"] = final_df[key_cols].astype(str).agg("|".join, axis=1)

    dups_df = final_df[final_df.duplicated(subset=["RECORD_KEY"], keep=False)].copy()
    dups_df["DUP_GROUP"] = dups_df.groupby("RECORD_KEY").ngroup() + 1
    dups_df["DUP_COUNT"] = dups_df.groupby("RECORD_KEY")["RECORD_KEY"].transform("count")

    output_path = fr"{path}\SAP_DS_ALL_TABLE_MAPPING_{timestamp}.csv"
    dups_path = fr"{path}\SAP_DS_TABLE_MAPPING_DUPLICATES_{timestamp}.csv"

    final_df.to_csv(output_path, index=False)
    dups_df.to_csv(dups_path, index=False)

    print(f"Done. Wrote: {output_path} | Rows: {len(final_df)}")


if __name__ == "__main__":
    main()
