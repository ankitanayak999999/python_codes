import time
from snowflake.snowpark.functions import (
    col, lit, trim, call_function,
    iff, sum as sf_sum, count_distinct,
    min as sf_min, max as sf_max
)

def profile_dataframe(session, df, table_name):

    start_total = time.time()

    df = df.select([col(c).alias(c.upper()) for c in df.columns])

    total_rows = df.count()
    denom = total_rows if total_rows else 1

    rows = []

    total_cols = len(df.schema.fields)
    col_index = 0

    for field in df.schema.fields:

        col_index += 1
        col_start = time.time()

        col_name = field.name
        col_type = field.datatype

        print(f"\nProcessing column {col_index}/{total_cols}: {col_name}")

        c = col(col_name)

        step = time.time()
        c_raw = trim(c.cast("STRING"))
        c1 = call_function("NULLIF", c_raw, lit(""))
        print(f"  Clean column time: {time.time() - step:.2f} sec")

        step = time.time()
        empty_str_cnt = df.filter(c.is_not_null() & (c_raw == lit(""))).count()
        print(f"  empty_str_cnt time: {time.time() - step:.2f} sec")

        step = time.time()
        null_cnt = df.filter(c.is_null()).count()
        print(f"  null_cnt time: {time.time() - step:.2f} sec")

        step = time.time()
        zero_cnt = df.filter(
            c1.is_not_null() &
            (call_function("TRY_TO_NUMBER", c1) == lit(0))
        ).count()
        print(f"  zero_cnt time: {time.time() - step:.2f} sec")

        step = time.time()
        num_cnt = df.filter(
            c1.is_not_null() &
            call_function("TRY_TO_NUMBER", c1).is_not_null()
        ).count()
        print(f"  numeric_cnt time: {time.time() - step:.2f} sec")

        step = time.time()
        date_cnt = df.filter(
            c1.is_not_null() &
            call_function("TRY_TO_DATE", c1).is_not_null()
        ).count()
        print(f"  date_cnt time: {time.time() - step:.2f} sec")

        step = time.time()
        distinct_cnt = df.select(c).distinct().count()
        print(f"  distinct_cnt time: {time.time() - step:.2f} sec")

        step = time.time()
        min_val = df.select(sf_min(c)).collect()[0][0]
        print(f"  min_val time: {time.time() - step:.2f} sec")

        step = time.time()
        max_val = df.select(sf_max(c)).collect()[0][0]
        print(f"  max_val time: {time.time() - step:.2f} sec")

        empty_str_pct = f"{(empty_str_cnt * 100.0 / denom):.4f} %"
        null_pct = f"{(null_cnt * 100.0 / denom):.4f} %"
        zero_pct = f"{(zero_cnt * 100.0 / denom):.4f} %"
        num_pct = f"{(num_cnt * 100.0 / denom):.4f} %"
        date_pct = f"{(date_cnt * 100.0 / denom):.4f} %"

        is_numeric = "Y" if (empty_str_cnt + null_cnt + num_cnt) == total_rows else "N"
        is_date = "Y" if (empty_str_cnt + null_cnt + date_cnt) == total_rows else "N"

        rows.append({
            "TABLE_NAME": table_name,
            "COLUMN_NAME": col_name,
            "DATA_TYPE": str(col_type),
            "TOTAL_ROWS": total_rows,

            "EMPTY_STR_CNT": empty_str_cnt,
            "EMPTY_STR_PCT": empty_str_pct,

            "NULL_COUNT": null_cnt,
            "NULL_PCT": null_pct,

            "ZERO_COUNT": zero_cnt,
            "ZERO_PCT": zero_pct,

            "NUMERIC_COUNT": num_cnt,
            "NUMERIC_PCT": num_pct,

            "DATE_COUNT": date_cnt,
            "DATE_PCT": date_pct,

            "IS_NUMERIC": is_numeric,
            "IS_DATE": is_date,

            "DISTINCT_COUNT": distinct_cnt,
            "MIN_VALUE": None if min_val is None else str(min_val),
            "MAX_VALUE": None if max_val is None else str(max_val)
        })

        print(f"Column {col_name} finished in {time.time() - col_start:.2f} sec")

    print(f"\nTOTAL EXECUTION TIME: {time.time() - start_total:.2f} sec")

    return session.create_dataframe(rows)
