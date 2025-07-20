def extract_all_zips(src_path):
    total_success = 0
    total_failures = 0
    failed_zips = []

    while True:
        zip_files = []
        for root, dirs, files in os.walk(src_path):
            for file in files:
                if file.lower().endswith(".zip"):
                    zip_files.append(os.path.join(root, file))

        if not zip_files:  # No more zip files
            break

        current_success = 0
        current_fail = 0

        for zip_path in zip_files:
            if not os.path.exists(zip_path):
                print(f"[MISSING] {zip_path}")
                failed_zips.append(zip_path)
                total_failures += 1
                current_fail += 1
                continue

            try:
                extract_to = os.path.splitext(zip_path)[0]
                os.makedirs(extract_to, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_to)
                os.remove(zip_path)
                total_success += 1
                current_success += 1
                print(f"[SUCCESS] {zip_path}")
            except Exception as e:
                print(f"[ERROR] {zip_path}: {e}")
                failed_zips.append(zip_path)
                total_failures += 1
                current_fail += 1

        print(f"\nIteration summary: {current_success} success, {current_fail} failed.\n")

        if current_success == 0:  # If nothing was extracted, break
            break

    # Save failed zips
    if failed_zips:
        log_file = os.path.join(src_path, "failed_zips.txt")
        with open(log_file, "w") as f:
            for z in failed_zips:
                f.write(z + "\n")
        print(f"\nFINAL SUMMARY: {total_success} success, {total_failures} failed (saved to {log_file})")
    else:
        print(f"\nFINAL SUMMARY: {total_success} success, {total_failures} failed")
