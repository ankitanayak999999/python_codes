def main():
    import os, glob, datetime
    import pandas as pd

    # --- timestamp ---
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")

    # (you had a REPO_LIST here; leaving as-is if you still want it)
    REPO_LIST = ['DSD_PROD_COPY_ODS', 'PROD_COPY_EDW_1']

    # --- keep your paths here (unchanged) ---
    # Set this to the repo folder you’re processing so REPO_NAME is useful
    path = r"C:\Users\raksahu\Downloads\python\input\sap_ds_xml_files\DSD_PROD_COPY_ODS"
    REPO_NAME = os.path.basename(path)
    single_file = fr"{path}\export_af.xml"

    # gather files
    all_files = glob.glob(os.path.join(path, "*.xml"))
    print(all_files)
    if single_file in all_files:
        all_files = [single_file]

    print(f"total number of files present in the path ({len(all_files)})")
    print(all_files)

    # parse each file
    df_list = []
    for i, file in enumerate(all_files):
        print(f"Row Number:{i}--{file}")
        df = parse_single_xml(file)
        df_list.append(df)

    final_df = pd.concat(df_list, ignore_index=True)

    # --- column rename map (patched to include the two new columns) ---
    rename_mapping = {
        'project_name'               : 'PROJECT_NAME',
        'job_name'                   : 'JOB_NAME',
        'dataflow_name'              : 'DATAFLOW_NAME',
        'role'                       : 'TRANFORMATION_TYPE',
        'datastore'                  : 'DATA_STORE',
        'schema'                     : 'SCHEMA_NAME',
        'table'                      : 'TABLE_NAME',
        'transformation_position'    : 'TRANSFORMATION_POSITION',
        'transformation_usage_count' : 'TRANSFORMATION_USAGE_COUNT',
        'source_line'                : 'SOURCE_LINE',
        'custom_sql_text'            : 'CUSTOM_SQL_TEXT',

        # <<< NEW >>>
        'datastore_details'          : 'DATASTORE_DETAILS',
        'sql_pick_method'            : 'SQL_PICK_METHOD',
    }

    # keep only these columns, in this order
    final_df = final_df.rename(columns=rename_mapping)[list(rename_mapping.values())]

    # --- keep your key at the end (first 7 columns) ---
    key_cols = ["PROJECT_NAME","JOB_NAME","DATAFLOW_NAME","TRANFORMATION_TYPE","DATA_STORE","SCHEMA_NAME","TABLE_NAME"]
    final_df["RECORD_KEY"] = final_df[key_cols].astype(str).agg("||".join, axis=1)
    final_df["SQL_LENGTH"] = final_df["CUSTOM_SQL_TEXT"].astype(str).apply(len)

    # carry repo folder name for traceability (you had this in your v17)
    final_df["REPO_NAME"] = REPO_NAME

    # sort so the “keep first” below keeps the longest SQL per key (unchanged)
    final_df = final_df.sort_values(["RECORD_KEY", "SQL_LENGTH"], ascending=[True, False])

    # duplicates snapshot (unchanged)
    dups_df = final_df[final_df.duplicated(subset=["RECORD_KEY"], keep=False)].copy()
    dups_df["DUP_GROUP"] = dups_df.groupby("RECORD_KEY").ngroup() + 1
    dups_df["DUP_COUNT"] = dups_df.groupby("RECORD_KEY")["RECORD_KEY"].transform("count")

    # keep first (longest SQL per key because of the sort above)
    final_df = final_df.drop_duplicates(subset="RECORD_KEY", keep="first").reset_index(drop=True)

    # outputs (unchanged names from your screenshot)
    output_path = fr"{path}\SAP_DS_TABLE_DEPENDENCIES_{timestamp}.csv"
    dups_path   = fr"{path}\SAP_DS_TABLE_DEPENDENCIES_DUPLICATES_{timestamp}.csv"

    final_df.to_csv(output_path, index=False)
    dups_df.to_csv(dups_path, index=False)

    print(f"Done. Wrote: {output_path}  |  Rows: {len(final_df)}")
    print(f"Number of duplicate records: {len(dups_df)}")
