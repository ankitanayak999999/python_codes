import os
import re
import pandas as pd
from datetime import datetime

def scan_excel(config_file_path, data_file_path, project_path):
    print("✅ Excel Scan process Started...")

    # Step 1: Load Config
    try:
        df_search_key = pd.read_excel(config_file_path, sheet_name='search_keys', keep_default_na=False)
        df_search_key.columns = df_search_key.columns.str.upper()

        df_search_col = pd.read_excel(config_file_path, sheet_name='search_columns', keep_default_na=False)
        df_search_col.columns = df_search_col.columns.str.upper()

        search_key_list_1 = df_search_key['SEARCH_KEY_1'].dropna().astype(str).tolist()
        search_key_list_2_raw = df_search_key['SEARCH_KEY_2'].dropna().astype(str).tolist()

        # Split comma-separated values in SEARCH_KEY_2
        search_key_list_2 = []
        for val in search_key_list_2_raw:
            search_key_list_2.extend([x.strip() for x in val.split(",") if x.strip()])

        columns_to_check = df_search_col['COLUMN_NAMES'].dropna().astype(str).tolist()

    except Exception as e:
        print(f"❌ Invalid Config file selection or format: {e}")
        return

    # Step 2: Load Data File
    try:
        df_data = pd.read_excel(data_file_path)
        df_data.columns = df_data.columns.str.upper()
        output_file_name = os.path.splitext(os.path.basename(data_file_path))[0]
        print(f"📊 Total Rows/Cols in input file: {df_data.shape}")
    except Exception as e:
        print(f"❌ Failed to read data file: {e}")
        return

    # Step 3: Validate or Fallback Column Scan
    if not columns_to_check:
        print("⚠️ No scan columns provided, defaulting to all columns")
        columns_to_check = df_data.columns.tolist()
    else:
        print(f"🔍 Scanning in columns: {columns_to_check}")

    # Step 4: Smart Matching Function
    def find_matches(row, key_list):
        found = set()
        for col in columns_to_check:
            cell_text = str(row.get(col, "")).lower()
            for key in key_list:
                key_lower = key.lower()
                pattern = re.compile(rf'(?<!_)\b{re.escape(key_lower)}\b(?!_)')
                if pattern.search(cell_text):
                    found.add(key)
        return ", ".join(sorted(found)), len(found)

    # Step 5: Apply Matching
    df_data["Matched_1"], df_data["Matched_Cnt_1"] = zip(*df_data.apply(lambda row: find_matches(row, search_key_list_1), axis=1))
    df_data["Matched_2"], df_data["Matched_Cnt_2"] = zip(*df_data.apply(lambda row: find_matches(row, search_key_list_2), axis=1))

    # Step 6: Write Output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(project_path, f"{output_file_name}_scanned_output_{timestamp}.xlsx")
    df_data.to_excel(output_file, index=False)
    print(f"✅ Output saved as: {output_file}")
