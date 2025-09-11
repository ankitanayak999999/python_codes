# ------- lookup_ext (FUNCTION_CALL ONLY — no DIExpression fallback) -------
if tag == "function_call" and lower(a.get("name","")) == "lookup_ext":
    proj, job, df = context_for(e)
    schema_out = schema_out_from_DISchema(e, pm, cur_schema)

    # read ONLY attributes on the Function Call (authoritative)
    # e.g. <FUNCTION_CALL name="lookup_ext" type="DI"
    #       tableDatastore="DS_ODS" tableOwner="BRIOVIEW" tableName="BANK_CMMT_TYPE_GL_XREF" .../>
    dsx  = (a.get("tabledatastore") or a.get("tableDatastore") or "").strip()
    schx = (a.get("tableowner")     or a.get("tableOwner")     or "").strip()
    tbx  = (a.get("tablename")      or a.get("tableName")      or "").strip()

    if dsx and tbx and schema_out:
        k = (proj, job, df, _norm_key(dsx), _norm_key(schx), _norm_key(tbx))
        remember_display(dsx, schx, tbx)
        lookup_ext_pos[k].add(schema_out)
        seen_ext_keys.add(k)  # so nothing else can add a dup for this key





import re

def clean_sql(sql_text):
    # Replace parameterized schema like ${G_Schema} or [${G_Schema}]
    sql_text = re.sub(r"\[\$\{[^}]+\}\]", "DUMMY_SCHEMA", sql_text)
    sql_text = re.sub(r"\$\{[^}]+\}", "DUMMY_SCHEMA", sql_text)
    # Remove remaining square brackets
    sql_text = re.sub(r"\[|\]", "", sql_text)
    # -------------------- main --------------------

def main():
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")

    # keep your paths here
    path = r"C:\Users\raksahu\Downloads\python\input\sap_ds_xml_files"
    single_file = fr"{path}\export_af.xml"

    all_files = glob.glob(os.path.join(path, "*.xml"))
    print(all_files)
    if single_file in all_files:
        all_files = [single_file]

    print(f"total number of files present in the path ({len(all_files)})")
    print(all_files)

    df_list = []
    for i, file in enumerate(all_files):
        print(f"Row Number:{i}--{file}")
        df = parse_single_xml(file)
        df_list.append(df)

    final_df = pd.concat(df_list, ignore_index=True)

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
        'datastore_details'          : 'DATASTORE_DETAILS',
        'sql_pick_method'            : 'SQL_PICK_METHOD',
    }

    # keep only these columns, in this order
    final_df = final_df.rename(columns=rename_mapping)[list(rename_mapping.values())]

    # ---- compute SQL_LENGTH for dedupe sort
    final_df["SQL_LENGTH"] = final_df["CUSTOM_SQL_TEXT"].astype(str).apply(len)

    # ---- prefer DS_MATCHED for custom_sql keys; otherwise fall back to longest SQL
    key_cols = ["PROJECT_NAME","JOB_NAME","DATAFLOW_NAME","TRANFORMATION_TYPE","DATA_STORE","SCHEMA_NAME","TABLE_NAME"]
    final_df["RECORD_KEY"] = final_df[key_cols].astype(str).agg("||".join, axis=1)
    final_df["SQL_PICK_PRIORITY"] = (final_df["SQL_PICK_METHOD"].eq("DS_MATCHED")).astype(int)

    final_df = final_df.sort_values(
        ["RECORD_KEY","SQL_PICK_PRIORITY","SQL_LENGTH"],
        ascending=[True, False, False]
    )

    # duplicates snapshot (before final de-dupe)
    dups_df = final_df[final_df.duplicated(subset=["RECORD_KEY"], keep=False)].copy()
    dups_df["DUP_GROUP"] = dups_df.groupby("RECORD_KEY").ngroup() + 1
    dups_df["DUP_COUNT"] = dups_df.groupby("RECORD_KEY")["RECORD_KEY"].transform("count")

    # keep first per key (prefers DS match, then longest SQL)
    final_df = final_df.drop_duplicates(subset="RECORD_KEY", keep="first").reset_index(drop=True)

    output_path = fr"{path}\SAP_DS_ALL_TABLE_MAPPING_{timestamp}.csv"
    dups_path   = fr"{path}\SAP_DS_TABLE_MAPPING_DUPLICATES_{timestamp}.csv"

    final_df.to_csv(output_path, index=False)
    dups_df.to_csv(dups_path, index=False)

    print(f"Done. Wrote: {output_path}  |  Rows: {len(final_df)}")
    print(f"Number of duplicate records: {len(dups_df)}")


if __name__ == "__main__":
    print("**** Process started at:", datetime.datetime.now())
    main()
    print("**** Process completed at:", datetime.datetime.now())







