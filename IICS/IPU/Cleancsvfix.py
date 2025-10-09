import io
import pandas as pd

# ---- config
METER_NAME_COL_IDX = 2  # 0-based index; column 3 in your file

def fix_bad_fields(fields, expected_cols, meter_idx=METER_NAME_COL_IDX):
    """
    Try to repair a malformed CSV row by merging overflow fields back into
    the 'Meter Name' column or padding missing fields.
    Returns a fixed list (length == expected_cols) or None if it can't fix.
    """
    n = len(fields)
    if n == expected_cols:
        return fields

    # Too many columns → merge overflow into Meter Name
    if n > expected_cols and meter_idx < expected_cols:
        extra = n - expected_cols
        merged = fields[:]
        merged[meter_idx] = ",".join(merged[meter_idx: meter_idx + extra + 1])
        repaired = merged[:meter_idx+1] + merged[meter_idx + extra + 1:]
        return repaired if len(repaired) == expected_cols else None

    # Too few columns → pad with empty strings
    if n < expected_cols:
        return fields + [""] * (expected_cols - n)

    return None  # could not fix safely


def read_csv_with_repairs(cleaned_csv, logger, file_name):
    """
    Read a CSV file-like object:
      - capture bad lines during pd.read_csv()
      - repair them (e.g. commas inside Meter Name)
      - merge repaired rows back with good rows
    Returns (df_all, bad_lines, unrepaired_rows)
    """
    bad_lines = []

    # 1) Peek header for schema and rewind
    header_line = cleaned_csv.readline()
    header_fields = header_line.rstrip("\n\r").split(",")
    expected_cols = len(header_fields)
    cleaned_csv.seek(0)

    # 2) Read good rows; capture bad rows
    def bad_line_handler(bad_line, file_name=file_name):
        bad_lines.append({"file": file_name, "fields": bad_line})
        return None  # skip for now

    df_good = pd.read_csv(
        cleaned_csv,
        sep=",",
        engine="python",
        quotechar='"',
        dtype=str,                 # keep text to avoid dtype errors
        on_bad_lines=bad_line_handler
    )

    # 3) Attempt to repair bad rows
    repaired_rows = []
    unrepaired_rows = []

    for i, rec in enumerate(bad_lines, start=1):
        fields = rec["fields"]
        fixed = fix_bad_fields(fields, expected_cols, METER_NAME_COL_IDX)
        if fixed is None:
            unrepaired_rows.append({
                "idx": i,
                "file": rec["file"],
                "fields": fields,
                "reason": f"Could not coerce to {expected_cols} columns"
            })
        else:
            repaired_rows.append(fixed)

    # 4) Convert repaired rows to DataFrame and merge
    if repaired_rows:
        df_fixed = pd.DataFrame(repaired_rows, columns=header_fields)
        df_all = pd.concat([df_good, df_fixed], ignore_index=True)
        logger.info(
            f"📌 {file_name}: merged {len(repaired_rows)} repaired row(s) "
            f"with {len(df_good)} good row(s). Total: {len(df_all)}"
        )
    else:
        df_all = df_good
        logger.info(f"✅ {file_name}: no rows needed repair. Total: {len(df_all)}")

    # 5) Log unrepaired rows if any
    if unrepaired_rows:
        for r in unrepaired_rows:
            logger.warning(
                f"\n❌ Unrepaired row in {r['file']}:\n"
                f"   Fields: {r['fields']}\n"
                f"   Reason: {r['reason']}\n"
            )

    return df_all, bad_lines, unrepaired_rows
