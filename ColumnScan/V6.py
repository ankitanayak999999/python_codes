def extract_all_zips(src_path):
    """
    Recursively extract all zip files from src_path and delete them.
    """
    zip_files = []

    # Step 1: Collect all zip files in current structure
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.lower().endswith(".zip"):
                zip_files.append(os.path.join(root, file))

    # Step 2: Extract and delete each zip file
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(zip_path))
            os.remove(zip_path)
            print(f"Extracted and deleted: {zip_path}")
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")

    # Step 3: If new zip files were extracted, repeat
    if zip_files:
        extract_all_zips(src_path)
