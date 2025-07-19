import os
import json
import zipfile

def extract_all_zips(src_path):
    """
    Recursively extract all zip files from src_path and delete them.
    """
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.lower().endswith(".zip"):
                zip_path = os.path.join(root, file)
                extract_to = root  # Extract in the same folder
                
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_to)
                    os.remove(zip_path)  # Delete the zip file after extraction
                    print(f"Extracted and deleted: {zip_path}")
                    
                    # After extracting, check the same folder again for nested zips
                    extract_all_zips(extract_to)
                    
                except Exception as e:
                    print(f"Error extracting {zip_path}: {e}")

def process_mttask_files(src_path, tgt_path):
    """
    Scan all folders for mtTask (or mtTask.json), format and save them to tgt_path.
    The saved file is renamed with _<frsGuid>.json.
    """
    os.makedirs(tgt_path, exist_ok=True)
    
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.lower() == "mttask" or file.lower() == "mttask.json":
                file_path = os.path.join(root, file)
                try:
                    # Load JSON content
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Get frsGuid
                    frs_guid = data.get("frsGuid", "NO_GUID")
                    
                    # Build target filename
                    folder_name = os.path.basename(root)
                    new_name = f"{folder_name}_{frs_guid}.json"
                    tgt_file_path = os.path.join(tgt_path, new_name)
                    
                    # Save formatted JSON
                    with open(tgt_file_path, "w", encoding="utf-8") as out_f:
                        json.dump(data, out_f, indent=4, sort_keys=True)
                    
                    print(f"Saved formatted JSON: {tgt_file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
