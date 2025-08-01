import os
import pandas as pd

def read_any_file(file_path, sheet_name=None, read_all_sheets=False):
    """
    Reads various file formats into Pandas DataFrame(s).
    
    Args:
        file_path (str): Path to the file.
        sheet_name (str): For Excel, the sheet name or index to load.
        read_all_sheets (bool): If True, returns dict of all sheets.
    
    Returns:
        pd.DataFrame or dict of DataFrames
    """
    ext = os.path.splitext(file_path.lower())[-1]

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
            df.columns = df.columns.str.upper()
            return df

        elif ext in [".xls", ".xlsx"]:
            if read_all_sheets:
                xls = pd.read_excel(file_path, sheet_name=None, dtype=str, keep_default_na=False)
                return {sheet: df.rename(columns=str.upper) for sheet, df in xls.items()}
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str, keep_default_na=False)
                df.columns = df.columns.str.upper()
                return df

        elif ext == ".json":
            df = pd.read_json(file_path)
            df.columns = df.columns.str.upper()
            return df

        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
            df.columns = df.columns.str.upper()
            return df

        elif ext == ".tsv":
            df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
            df.columns = df.columns.str.upper()
            return df

        else:
            raise ValueError(f"❌ Unsupported file format: {ext}")

    except Exception as e:
        print(f"❌ Error reading file '{file_path}': {e}")
        return pd.DataFrame()  # Return empty DataFrame for fail-safewas 
