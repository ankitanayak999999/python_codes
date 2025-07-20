def extract_all_zips(src_path):
    zip_files = []
    sucess_cnt = 0
    error_cnt = 0
    failed_zips = []

    while True:
        # Step 1: Collect all zip files in the current directory tree
        zip_files.clear()
        for root, dirs, files in os.walk(src_path):
            for file in files:
                if file.lower().endswith(".zip"):
                    zip_files.append(os.path.join(root, file))

        if not zip_files:  # No more zip files found
            break

        new_success = 0
        new_errors = 0

        # Step 2: Extract each zip file into a folder named after the zip
        for zip_path in zip_files:
            try:
                extract_to = os.path.splitext(zip_path)[0]  # Remove .zip to make folder
                os.makedirs(extract_to, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_to)
                os.remove(zip_path)  # Delete the original zip file
                print(f"Extracted to {extract_to} and deleted: {zip_path}")
                sucess_cnt += 1
                new_success += 1
            except Exception as e:
                print(f"Error serial number {error_cnt}: {e}")
                error_cnt += 1
                new_errors += 1
                failed_zips.append(zip_path)

        if new_success == 0:  # If no zip files were successfully extracted, break
            break

    # Step 3: Save failed zip paths to file
    if failed_zips:
        cm.text_file_save(failed_zips)

    print(f"extract_all completed with success count: {sucess_cnt} and error count: {error_cnt}")
    if failed_zips:
        print("Failed to extract the following files:")
        for f in failed_zips:
            print("  ", f)the 
