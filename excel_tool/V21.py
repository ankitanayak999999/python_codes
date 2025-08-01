try:
    # Rename and move Config file
    new_config_name = f"{os.path.splitext(os.path.basename(Config_File_Path))[0]}_{timestamp}.xlsx"
    shutil.move(Config_File_Path, os.path.join(project_path, new_config_name))

    # Rename and move Data file
    ext = os.path.splitext(data_file_path)[1]
    new_data_name = f"{os.path.splitext(os.path.basename(data_file_path))[0]}_{timestamp}{ext}"
    shutil.move(data_file_path, os.path.join(project_path, new_data_name))

    print("📂 Input files moved and renamed to project folder.")
except Exception as e:
    print(f"⚠️ Failed to move input files: {e}")
