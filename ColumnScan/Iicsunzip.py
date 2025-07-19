import os
import re
import shutil
import zipfile
import json
import datetime
import pandas as pd  # <-- Added pandas

# Paths
SRC_PATH = "C:/Users/raksahu/Downloads/python/input"
TGT_PATH = "C:/Users/raksahu/Downloads/python/output"
LOG_FILE = os.path.join(TGT_PATH, "unzip_log.txt")
CSV_FILE = os.path.join(TGT_PATH, "frsguid_paths.csv")
MERGED_JSON_FILE = os.path.join(TGT_PATH, "merged_mttasks.json")

# Collect CSV rows
CSV_ROWS = []

def log_message(message):
    """Log messages to both console and log file."""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(message + "\n")

def copy_json(file_path, root):
    """
    Copy mtTask.json to target folder, appending _frsGuid to filename to avoid duplicates.
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
        project_path = root.split("Explore", 1)[-1].lstrip("\\/")

        # Add row for CSV
        CSV_ROWS.append({
            "frsGuid": frs_guid,
            "project_path": project_path,
            "full_path": file_path,
            "folder_name": folder_name
        })

        log_message(f"Copied JSON: {file_path} -> {tgt_file}")
    except Exception as e:
        log_message(f"Error reading {file_path}: {e}")

def extract_zip(zip_path, extract_to):
    """Extract a .zip file and delete it after extraction."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_obj:
            zip_obj.extractall(extract_to)
        log_message(f"Extracted ZIP: {zip_path} -> {extract_to}")
        os.remove(zip_path)
        log_message(f"Deleted ZIP: {zip_path}")
    except Exception as e:
        log_message(f"Error extracting {zip_path}: {e}")

def unzip_folder(src_path, tgt_path):
    """Recursively walk through the src_path, extracting ZIP files and copying mtTask.json."""
    for root, dirs, files in os.walk(src_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file == "mtTask.json":
                copy_json(file_path, root)
            if re.search(r"\.zip$", file, re.IGNORECASE):
                extract_to = file_path[:-4]
                if not os.path.exists(extract_to):
                    os.makedirs(extract_to)
                extract_zip(file_path, extract_to)
                unzip_folder(extract_to, tgt_path)

def save_csv_with_pandas(csv_file):
    """Save collected CSV rows to file using pandas DataFrame."""
    if CSV_ROWS:
        df = pd.DataFrame(CSV_ROWS)
        df.to_csv(csv_file, index=False)
        log_message(f"CSV file saved with pandas: {csv_file}")

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
                log_message(f"Merged JSON file: {file}")
            except Exception as e:
                log_message(f"Error merging {file}: {e}")

    with open(merged_file, 'w', encoding='utf-8') as out_file:
        json.dump(merged_data, out_file, indent=4, sort_keys=True)
    log_message(f"Merged JSON saved to: {merged_file}")

if __name__ == "__main__":
    os.makedirs(TGT_PATH, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write(f"**** Process started at: {datetime.datetime.now()} ****\n")

    log_message(f"Processing started at {datetime.datetime.now()}")
    unzip_folder(SRC_PATH, TGT_PATH)
    merge_json_files(TGT_PATH, MERGED_JSON_FILE)
    save_csv_with_pandas(CSV_FILE)
    log_message(f"Processing completed at {datetime.datetime.now()}")
