import tkinter as tk
from tkinter import filedialog

def get_inputs_files_and_strings():
    def browse_file1():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        entry_file1.delete(0, tk.END)
        entry_file1.insert(0, file_path)

    def browse_file2():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        entry_file2.delete(0, tk.END)
        entry_file2.insert(0, file_path)

    def submit():
        nonlocal result
        result = (
            entry_file1.get(),
            entry_file2.get(),
            entry_join_key.get(),
            entry_process_name.get(),
            entry_extra.get()
        )
        root.destroy()

    result = None
    root = tk.Tk()
    root.title("Provide Inputs")

    # File 1
    tk.Label(root, text="Select File 1:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    entry_file1 = tk.Entry(root, width=50)
    entry_file1.grid(row=0, column=1, padx=5)
    tk.Button(root, text="Browse", command=browse_file1).grid(row=0, column=2)

    # File 2
    tk.Label(root, text="Select File 2:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    entry_file2 = tk.Entry(root, width=50)
    entry_file2.grid(row=1, column=1, padx=5)
    tk.Button(root, text="Browse", command=browse_file2).grid(row=1, column=2)

    # Join Key
    tk.Label(root, text="Join Key:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    entry_join_key = tk.Entry(root, width=50)
    entry_join_key.grid(row=2, column=1, padx=5)

    # Process Name
    tk.Label(root, text="Process Name:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
    entry_process_name = tk.Entry(root, width=50)
    entry_process_name.grid(row=3, column=1, padx=5)

    # Extra Input
    tk.Label(root, text="Extra Input:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
    entry_extra = tk.Entry(root, width=50)
    entry_extra.grid(row=4, column=1, padx=5)

    # Submit Button
    tk.Button(root, text="Submit", command=submit).grid(row=5, column=0, columnspan=3, pady=10)

    root.mainloop()
    return result
