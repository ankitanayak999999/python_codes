# ============================================================
# FAST Snowpark profiling (same output column names as your tool)
# Key change: ONE aggregation query for ALL columns (plus 1 count)
# This reduces 900-col profiling from ~45 min to typically minutes.
# ============================================================

import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
from snowflake.snowpark.functions import (
    col, lit, trim, nullif, call_function,
    iff, sum as sf_sum, count_distinct,
    min as sf_min, max as sf_max
)

# ------------------------------------------------------------
# DO NOT change this signature if you already call it
# ------------------------------------------------------------
def profile_dataframe(session: Session, df, table_name: str):

    # Keep your behavior: uppercase columns
    df = df.select([col(c).alias(c.upper()) for c in df.columns])

    # 1 query
    total_rows = df.count()
    denom = total_rows if total_rows else 1

    agg_exprs = []
    meta = []  # (col_name, datatype)

    # Build aggregation expressions for ALL columns
    for field in df.schema.fields:
        col_name = field.name
        col_type = field.datatype
        c = col(col_name)

        # String-normalized value used for numeric/date checks
        # - cast to STRING, trim spaces
        # - convert '' to NULL
        c_raw = trim(c.cast("STRING"))
        c1 = nullif(c_raw, lit(""))

        # EMPTY_STR_CNT: non-null but trims to ''
        empty_cnt_expr = sf_sum(
            iff((c.is_not_null()) & (c_raw == lit("")), lit(1), lit(0))
        ).alias(f"{col_name}__EMPTY_STR_CNT")

        # NULL_COUNT: true NULLs OR blanks after nullif() (NOTE: this matches your use of c1 elsewhere)
        # If you want strictly "true NULL only", use (c.is_null()) instead.
        null_cnt_expr = sf_sum(
            iff(c.is_null(), lit(1), lit(0))
        ).alias(f"{col_name}__NULL_COUNT")

        # ZERO_COUNT: true numeric zero values (handles '0' too via TRY_TO_NUMBER)
        zero_cnt_expr = sf_sum(
            iff(
                c1.is_not_null() &
                (call_function("TRY_TO_NUMBER", c1) == lit(0)),
                lit(1), lit(0)
            )
        ).alias(f"{col_name}__ZERO_COUNT")

        # NUMERIC_COUNT: strict numeric convert (NULL/blank excluded by c1.is_not_null)
        numeric_cnt_expr = sf_sum(
            iff(
                c1.is_not_null() &
                call_function("TRY_TO_NUMBER", c1).is_not_null(),
                lit(1), lit(0)
            )
        ).alias(f"{col_name}__NUMERIC_COUNT")

        # DATE_COUNT: avoid 1970 false positives by requiring '-' or '/' before TRY_TO_DATE
        has_sep = call_function("REGEXP_LIKE", c1, lit(r".*[-/].*"))
        date_cnt_expr = sf_sum(
            iff(
                c1.is_not_null() &
                has_sep &
                call_function("TRY_TO_DATE", c1).is_not_null(),
                lit(1), lit(0)
            )
        ).alias(f"{col_name}__DATE_COUNT")

        # DISTINCT_COUNT / MIN_VALUE / MAX_VALUE (exact, may be heavy on huge tables)
        distinct_expr = count_distinct(c).alias(f"{col_name}__DISTINCT_COUNT")
        min_expr = sf_min(c).alias(f"{col_name}__MIN_VALUE")
        max_expr = sf_max(c).alias(f"{col_name}__MAX_VALUE")

        agg_exprs.extend([
            empty_cnt_expr, null_cnt_expr, zero_cnt_expr,
            numeric_cnt_expr, date_cnt_expr,
            distinct_expr, min_expr, max_expr
        ])
        meta.append((col_name, col_type))

    # 1 query for all metrics across all columns
    agg_row = df.agg(*agg_exprs).collect()[0]

    def pct_fmt(x: int) -> str:
        return f"{(x * 100.0 / denom):.4f} %"

    rows = []
    for (col_name, col_type) in meta:
        empty_str_cnt = int(agg_row[f"{col_name}__EMPTY_STR_CNT"] or 0)
        null_cnt      = int(agg_row[f"{col_name}__NULL_COUNT"] or 0)
        zero_cnt      = int(agg_row[f"{col_name}__ZERO_COUNT"] or 0)
        num_cnt       = int(agg_row[f"{col_name}__NUMERIC_COUNT"] or 0)
        date_cnt      = int(agg_row[f"{col_name}__DATE_COUNT"] or 0)
        distinct_cnt  = int(agg_row[f"{col_name}__DISTINCT_COUNT"] or 0)

        min_val_raw = agg_row[f"{col_name}__MIN_VALUE"]
        max_val_raw = agg_row[f"{col_name}__MAX_VALUE"]
        min_val = None if min_val_raw is None else str(min_val_raw)
        max_val = None if max_val_raw is None else str(max_val_raw)

        # Keep your % columns as strings with '%'
        empty_str_pct_formatted = pct_fmt(empty_str_cnt)
        null_pct_formatted      = pct_fmt(null_cnt)
        zero_pct_formatted      = pct_fmt(zero_cnt)
        num_pct_formatted       = pct_fmt(num_cnt)
        date_pct_formatted      = pct_fmt(date_cnt)

        # Your rule:
        # If EMPTY + NULL + NUMERIC == TOTAL => Y else N
        is_numeric = "Y" if (empty_str_cnt + null_cnt + num_cnt) == total_rows else "N"
        is_date    = "Y" if (empty_str_cnt + null_cnt + date_cnt) == total_rows else "N"

        rows.append({
            "TABLE_NAME": table_name,
            "COLUMN_NAME": col_name,
            "DATA_TYPE": str(col_type),
            "TOTAL_ROWS": total_rows,

            "EMPTY_STR_CNT": empty_str_cnt,
            "EMPTY_STR_PCT": empty_str_pct_formatted,

            "NULL_COUNT": null_cnt,
            "NULL_PCT": null_pct_formatted,

            "ZERO_COUNT": zero_cnt,
            "ZERO_PCT": zero_pct_formatted,

            "NUMERIC_COUNT": num_cnt,
            "NUMERIC_PCT": num_pct_formatted,

            "DATE_COUNT": date_cnt,
            "DATE_PCT": date_pct_formatted,

            "IS_NUMERIC": is_numeric,
            "IS_DATE": is_date,

            "DISTINCT_COUNT": distinct_cnt,
            "MIN_VALUE": min_val,
            "MAX_VALUE": max_val
        })

    return session.create_dataframe(rows)


# ------------------------------------------------------------
# Example main(session) for Snowflake Worksheet
# Keep your variable names & flow if you already have them.
# ------------------------------------------------------------
def main(session: Session):

    # Example: replace with your existing TABLE_NAMES / WHERE_CLAUSE logic
    TABLE_NAMES = [
        "ENTP_DEV_DL_DB.ENTP_PL_SCH.PL_LOAN_ACCOUNT"
    ]

    WHERE_CLAUSE = "1=1"  # or your filter string

    final_df = None

    for t in TABLE_NAMES:
        df = session.table(t).filter(WHERE_CLAUSE)
        prof = profile_dataframe(session, df, t)

        final_df = prof if final_df is None else final_df.union_all(prof)

    return final_df
