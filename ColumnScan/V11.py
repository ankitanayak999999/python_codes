import os
import zipfile

def extract_all_zips(src_path, project_path):
    total_scanned = 0
    total_success = 0
    total_failures = 0
    failed_zips = []
    failed_set = set()  # Keep track of failed ZIPs

    while True:
        zip_files = []
        for root, dirs, files in os.walk(src_path):
            for file in files:
                if file.lower().endswith(".zip"):
                    file_path = os.path.join(root, file)
                    if file_path not in failed_set:  # Skip already failed files
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
            zip_path = os.path.normpath(zip_path)
            if os.name == 'nt':
                zip_path = "\\\\?\\" + zip_path  # Enable long path support

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

        if current_success == 0:  # No success in this pass, exit to avoid infinite loop
            break

    # Save failed zips
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
