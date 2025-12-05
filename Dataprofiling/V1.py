def profile_dataframe(session, df, table_name: str, columns_to_profile=None):
    """
    columns_to_profile = list of column names to include.
    If None → profile all columns.
    """

    total_rows = df.count()

    if columns_to_profile:
        # enforce uppercase since Snowflake columns are uppercase by default
        columns_to_profile = [c.upper() for c in columns_to_profile]
    else:
        # fall back to all columns
        columns_to_profile = [f.name for f in df.schema.fields]

    rows = []

    for field in df.schema.fields:
        col_name = field.name

        # skip columns not requested
        if col_name not in columns_to_profile:
            continue

        col_type = field.datatype
        c = col(col_name)

        # --------------------
        # SAME LOGIC AS BEFORE
        # --------------------
        non_null_cnt = df.filter(c.is_not_null()).count()
        null_cnt = total_rows - non_null_cnt
        null_pct = (null_cnt * 100.0) / total_rows

        distinct_cnt = df.select(c).distinct().count()

        zero_cnt = None
        zero_pct = None
        if is_numeric_type(col_type):
            zero_cnt = df.filter((c == 0) | (c == 0.0)).count()
            zero_pct = (zero_cnt * 100.0) / total_rows

        min_max = (
            df.select(
                sf_min(c).cast("STRING").alias("MIN_VALUE"),
                sf_max(c).cast("STRING").alias("MAX_VALUE")
            ).collect()[0]
        )

        min_val = min_max["MIN_VALUE"]
        max_val = min_max["MAX_VALUE"]

        avg_len = None
        if isinstance(col_type, (T.StringType, T.VariantType)):
            avg_len = df.select(length(c)).agg({"LEN": "avg"}).collect()[0][0]

        rows.append({
            "TABLE_NAME": table_name,
            "COLUMN_NAME": col_name,
            "DATA_TYPE": str(col_type),
            "ROW_COUNT": total_rows,
            "NON_NULL_COUNT": non_null_cnt,
            "NULL_COUNT": null_cnt,
            "NULL_PCT": null_pct,
            "DISTINCT_COUNT": distinct_cnt,
            "ZERO_COUNT": zero_cnt,
            "ZERO_PCT": zero_pct,
            "MIN_VALUE": min_val,
            "MAX_VALUE": max_val,
            "AVG_LENGTH": avg_len,
        })

    return session.create_dataframe()
     
  
def main(session):
    table_name = "ENTP_DEV_DL_DB.ENTP_PL_SCH.PL_TREASURY_REJECTS"

    df = session.table(table_name)

    # filter rows (optional)
    df_filtered = df.filter("AS_OF_DATE = '2024-11-30'")

    # choose specific columns
    columns_to_profile = [
        "EXTERNAL_IDENTIFIER",
        "EXTERNAL_IDENTIFIER_HASH",
        "AMOUNT"
    ]

    result = profile_dataframe(
        session,
        df_filtered,
        table_name,
        columns_to_profile=columns_to_profile
    )

    return result

