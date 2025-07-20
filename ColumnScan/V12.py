import os
import json
import zipfile
import pandas as pd
import datetime
import re
import sys
import time
import cm as cm  # Your custom module

sys.path.append('C:/Users/raksahu/OneDrive - City National Bank/Documents/VS_Code')

# -------------------------------
# Extract all ZIP files
# -------------------------------
def extract_all_zips(src_path, project_path):
    total_scanned = 0
    total_success = 0
    total_failures = 0
    failed_zips = []
    failed_set = set()

    # Add \\?\ prefix once for Windows long paths
    if os.name == 'nt' and not src_path.startswith('\\\\?\\'):
        src_path = '\\\\?\\' + os.path.normpath(src_path)

    while True:
        zip_files = []
        for root, dirs, files in os.walk(src_path):
            for file in files:
                if file.lower().endswith(".zip"):
                    file_path = os.path.normpath(os.path.join(root, file))
                    if file_path not in failed_set:
                        zip_files.append(file_path)

        if not zip_files:
            if total_scanned == 0:
                print("No zip files found.")
            break

        total_scanned += len(zip_files)
        print(f"\nFound {len(zip_files)} zip files to extract in this pass...")

        current_success = 0
        current_fail = 0

        for zip_path in zip_files:
            print(f"Scanning: {zip_path}")
            try:
                extract_to = os.path.splitext(zip_path)[0]
                os.makedirs(extract_to, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_to)
                os.remove(zip_path)
                total_success += 1
                current_success += 1
                print(f"[SUCCESS] Extracted: {zip_path}")
            except Exception as e:
                print(f"[ERROR] {zip_path}: {e}")
                failed_set.add(zip_path)
                failed_zips.append(zip_path)
                total_failures += 1
                current_fail += 1

        print(f"Pass Summary: {current_success} extracted, {current_fail} failed.")
        if current_success == 0:
            break

    if failed_zips:
        log_file = os.path.join(project_path, "failed_zips.txt")
        with open(log_file, "w") as f:
            for z in failed_zips:
                f.write(z + "\n")
        print(f"\nFINAL SUMMARY:")
        print(f"  Total ZIP files scanned: {total_scanned}")
        print(f"  Extracted successfully:  {total_success}")
        print(f"  Failed to extract:       {total_failures}")
        print(f"  Failed list saved to: {log_file}")
    else:
        print(f"\nFINAL SUMMARY:")
        print(f"  Total ZIP files scanned: {total_scanned}")
        print(f"  Extracted successfully:  {total_success}")
        print(f"  Failed to extract:       {total_failures}")


# -------------------------------
# Extract MTT JSON files
# -------------------------------
def extract_mtt_json_files(src_path, tgt_path, project_path):
    """
    Scan all folders for mtTask (or mtTask.json), format and save them to tgt_path.
    The saved file is renamed with <frsGuid>.json.
    """
    os.makedirs(tgt_path, exist_ok=True)

    # Add \\?\ prefix once for Windows long paths
    if os.name == 'nt' and not src_path.startswith('\\\\?\\'):
        src_path = '\\\\?\\' + os.path.normpath(src_path)

    total_found = 0
    total_saved = 0
    total_failed = 0
    merged_data = []
    csv_rows = []
    guid_sets = set()
    duplicate_files = []

    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.lower() == "mttask.json":
                file_path = os.path.normpath(os.path.join(root, file))
                root_path = os.path.normpath(root)

                total_found += 1
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if isinstance(data, dict):
                        json_data = data
                    elif isinstance(data, list) and isinstance(data[0], dict):
                        json_data = data[0]
                    else:
                        print(f"Invalid json file {file_path}")
                        continue

                    frs_guid = json_data.get("frsGuid", "NO_GUID")
                    if frs_guid in guid_sets:
                        duplicate_files.append(file_path)
                        continue
                    guid_sets.add(frs_guid)

                    folder_name = os.path.basename(root)
                    new_name = f"{folder_name}_{frs_guid}.json"
                    tgt_file_path = os.path.join(tgt_path, new_name)

                    with open(tgt_file_path, "w", encoding="utf-8") as out_f:
                        json.dump(data, out_f, indent=4, sort_keys=True)

                    merged_data.append(json_data)

                    root_path_rel = os.path.relpath(root_path, start=src_path)
                    export_file_name = root_path_rel.split(r"[\\/]", 1)[0]
                    export_timestamp_str = export_file_name.split("-")[-1]
                    export_timestamp = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(int(export_timestamp_str) / 1000)
                    )

                    obj_path = root_path_rel.split("Explore", 1)[-1].lstrip("\\/")
                    obj_name = re.split(r"[\\/]", obj_path)[-1]
                    obj_project_name = re.split(r"[\\/]", obj_path)[0]

                    csv_rows.append({
                        "OBJ_PROJECT_NAME": obj_project_name,
                        "OBJ_NAME": obj_name,
                        "OBJ_ID": frs_guid,
                        "OBJ_TYPE": "MTT",
                        "OBJ_PATH": obj_path,
                        "OBJ_UPDATEDBY": "Rakesh Sahu",
                        "OBJ_UPDATEDTIME": export_timestamp
                    })

                    total_saved += 1

                except json.JSONDecodeError:
                    total_failed += 1
                    print(f"Invalid JSON File: {file_path}")
                except Exception as e:
                    total_failed += 1
                    print(f"Error processing {file_path}: {e}")

    print(f"\nTotal MTT found: {total_found}")
    print(f"Total MTT saved: {total_saved}")
    print(f"Total MTT failed: {total_failed}")
    print(f"Total duplicate files: {len(duplicate_files)}")

    dup_file = os.path.join(project_path, "duplicate_files.txt")
    with open(dup_file, "w") as f:
        for z in duplicate_files:
            f.write(z + "\n")

    merged_file_name = os.path.join(project_path, "all_mtt_json.txt")
    with open(merged_file_name, "w", encoding="utf-8") as out_f:
        json.dump(merged_data, out_f, indent=4)
    print(f"Merged JSON file saved at {merged_file_name} with {len(merged_data)} records.")

    csv_file_name = os.path.join(project_path, "all_mtt_location.csv")
    result_df = pd.DataFrame(csv_rows)
    result_df.to_csv(csv_file_name, index=False)
    print(f"CSV file saved at {csv_file_name} with {len(result_df)} rows.")


# -------------------------------
# Main Runner (UNCHANGED)
# -------------------------------
def main_run():
    project_path = "C:/Users/raksahu/Downloads/python/iics_metadata"
    src_path = "C:/Users/raksahu/Downloads/python/iics_metadata/export_files"
    tgt_path = "C:/Users/raksahu/Downloads/python/iics_metadata/json_files"
    # extract_all_zips(src_path, project_path)
    extract_mtt_json_files(src_path, tgt_path, project_path)


if __name__ == "__main__":
    print("*** Process started at:", datetime.datetime.now())
    main_run()
    print("*** Process completed at:", datetime.datetime.now())
