import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import date

def get_user_inputs():
    from tkinter import filedialog, messagebox

    result = None
    root = tk.Tk()
    root.title("Provide Inputs")
    root.geometry("800x300")

    # File 1
    tk.Label(root, text="Select File 1 *", fg="red").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    entry_file1 = tk.Entry(root, width=70)
    entry_file1.grid(row=0, column=1, padx=5)
    def browse_file1():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        entry_file1.delete(0, tk.END)
        entry_file1.insert(0, file_path)
    tk.Button(root, text="Browse", command=browse_file1).grid(row=0, column=2, padx=(0,10))

    # File 2
    tk.Label(root, text="Select File 2 *", fg="red").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    entry_file2 = tk.Entry(root, width=70)
    entry_file2.grid(row=1, column=1, padx=5)
    def browse_file2():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        entry_file2.delete(0, tk.END)
        entry_file2.insert(0, file_path)
    tk.Button(root, text="Browse", command=browse_file2).grid(row=1, column=2, padx=(0,10))

    # Unique Key
    tk.Label(root, text="Enter Unique ID to Compare *", fg="red").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    entry_unique_key = tk.Entry(root, width=70)
    entry_unique_key.grid(row=2, column=1, padx=5)

    # Result File Name
    tk.Label(root, text="Enter Result File Name").grid(row=3, column=0, padx=5, pady=5, sticky="w")
    entry_result_file_name = tk.Entry(root, width=70)
    entry_result_file_name.grid(row=3, column=1, padx=5)

    # File 1 Suffix
    tk.Label(root, text="Enter File 1 Suffix").grid(row=4, column=0, padx=5, pady=5, sticky="w")
    entry_file1_suffix = tk.Entry(root, width=70)
    entry_file1_suffix.grid(row=4, column=1, padx=5)

    # File 2 Suffix
    tk.Label(root, text="Enter File 2 Suffix").grid(row=5, column=0, padx=5, pady=5, sticky="w")
    entry_file2_suffix = tk.Entry(root, width=70)
    entry_file2_suffix.grid(row=5, column=1, padx=5)

    # Focus on first field
    entry_file1.focus()

    def submit():
        nonlocal result
        if not entry_file1.get() or not entry_file2.get() or not entry_unique_key.get():
            messagebox.showerror("Missing Input", "Please fill in all required fields marked with *")
            return
        result = (
            entry_file1.get(),
            entry_file2.get(),
            entry_unique_key.get(),
            entry_result_file_name.get(),
            entry_file1_suffix.get(),
            entry_file2_suffix.get()
        )
        root.destroy()

    # Submit Button
    tk.Button(root, text="Submit", command=submit).grid(row=6, column=0, columnspan=3, pady=20)

    root.mainloop()
    return result
