import os
import re
import shutil
import zipfile
import json
import pandas as pd

# Paths
PROJECT_PATH = os.path.normpath("C:/Users/raksahu/Downloads/python/iics_metadata")
SRC_PATH = os.path.normpath("C:/Users/raksahu/Downloads/python/iics_metadata/export_files")
TGT_PATH = os.path.normpath("C:/Users/raksahu/Downloads/python/iics_metadata/json_files")

CSV_FILE = os.path.join(PROJECT_PATH, "MTT_LOCATIONS.csv")
MERGED_JSON_FILE = os.path.join(PROJECT_PATH, "merged_mttasks.json")

# Create target folder
os.makedirs(TGT_PATH, exist_ok=True)

# Collect CSV rows
CSV_ROWS = []

def copy_json(file_path, root):
    """
    Copy mtTask or mtTask.json to target folder, appending _frsGuid to filename to avoid duplicates.
    Also store frsGuid, project path, full path, and folder name for CSV.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        frs_guid = data.get('frsGuid', 'NO_GUID')

        folder_name = os.path.basename(root)
        new_name = f"{folder_name}_{frs_guid}.json"

        tgt_file = os.path.join(TGT_PATH, new_name)

        # Save formatted JSON
        with open(tgt_file, 'w', encoding='utf-8') as out_file:
            json.dump(data, out_file, indent=4, sort_keys=True)

        # Capture project path after 'Explore'
        normalized_root = os.path.normpath(root)
        if "Explore" in normalized_root:
            project_path = normalized_root.split("Explore", 1)[-1].lstrip(os.sep)
        else:
            project_path = ""

        # Add row for CSV
        CSV_ROWS.append({
            "frsGuid": frs_guid,
            "project_path": project_path,
            "full_path": file_path,
            "folder_name": folder_name
        })
    except Exception as e:
        print(f"Error copying {file_path}: {e}")

def extract_zip(zip_path, extract_to):
    """Extract a .zip file and delete it after extraction."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_obj:
            zip_obj.extractall(extract_to)
        os.remove(zip_path)
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")

def unzip_folder(src_path, tgt_path):
    """Recursively walk through the src_path, extracting ZIP files and copying mtTask files."""
    for root, dirs, files in os.walk(src_path):
        root = os.path.normpath(root)  # Normalize root path
        for file in files:
            file_path = os.path.normpath(os.path.join(root, file))
            # Check for mtTask files (with or without .json)
            if file.lower() == "mttask" or file.lower() == "mttask.json":
                print(f"Found mtTask: {file_path}")
                copy_json(file_path, root)
            # Check for ZIP files
            if re.search(r"\.zip$", file, re.IGNORECASE):
                extract_to = os.path.normpath(file_path[:-4])
                if not os.path.exists(extract_to):
                    os.makedirs(extract_to)
                extract_zip(file_path, extract_to)
                unzip_folder(extract_to, tgt_path)

def save_csv_with_pandas(csv_file):
    """Save collected CSV rows to file using pandas DataFrame."""
    if CSV_ROWS:
        df = pd.DataFrame(CSV_ROWS)
        df.to_csv(csv_file, index=False)
        print(f"CSV file saved: {csv_file}")

def merge_json_files(output_folder, merged_file):
    """Merge all JSON files in the output_folder into a single formatted JSON list."""
    merged_data = []
    for file in os.listdir(output_folder):
        if file.endswith(".json"):
            file_path = os.path.join(output_folder, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    merged_data.append(data)
            except Exception as e:
                print(f"Error merging {file}: {e}")

    with open(merged_file, 'w', encoding='utf-8') as out_file:
        json.dump(merged_data, out_file, indent=4, sort_keys=True)
    print(f"Merged JSON saved: {merged_file}")

if __name__ == "__main__":
    unzip_folder(SRC_PATH, TGT_PATH)
    merge_json_files(TGT_PATH, MERGED_JSON_FILE)
    save_csv_with_pandas(CSV_FILE)
