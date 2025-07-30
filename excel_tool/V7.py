import tkinter as tk
from tkinter import filedialog, messagebox

def get_user_inputs():
    result = None
    root = tk.Tk()
    root.title("Provide Inputs")
    root.geometry("800x320")

    # Helper: move to next field on Enter
    def focus_next(event):
        event.widget.tk_focusNext().focus()
        return "break"

    # File 1
    tk.Label(root, text="Select File 1 *", fg="red").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    entry_file1 = tk.Entry(root, width=70)
    entry_file1.grid(row=0, column=1, padx=5)
    warning_file1 = tk.Label(root, text="", fg="red")
    warning_file1.grid(row=0, column=2, sticky="w")
    def browse_file1():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        entry_file1.delete(0, tk.END)
        entry_file1.insert(0, file_path)
    tk.Button(root, text="Browse", command=browse_file1).grid(row=0, column=3, padx=(0, 10))

    # File 2
    tk.Label(root, text="Select File 2 *", fg="red").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    entry_file2 = tk.Entry(root, width=70)
    entry_file2.grid(row=1, column=1, padx=5)
    warning_file2 = tk.Label(root, text="", fg="red")
    warning_file2.grid(row=1, column=2, sticky="w")
    def browse_file2():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        entry_file2.delete(0, tk.END)
        entry_file2.insert(0, file_path)
    tk.Button(root, text="Browse", command=browse_file2).grid(row=1, column=3, padx=(0, 10))

    # Unique Key
    tk.Label(root, text="Enter Unique ID to Compare *", fg="red").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    entry_unique_key = tk.Entry(root, width=70)
    entry_unique_key.grid(row=2, column=1, padx=5)
    warning_key = tk.Label(root, text="", fg="red")
    warning_key.grid(row=2, column=2, sticky="w")

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

    # Submit logic
    def submit():
        nonlocal result

        # Clear previous warnings
        warning_file1.config(text="")
        warning_file2.config(text="")
        warning_key.config(text="")

        # Required field validation
        missing = False
        if not entry_file1.get():
            warning_file1.config(text="Required")
            missing = True
        if not entry_file2.get():
            warning_file2.config(text="Required")
            missing = True
        if not entry_unique_key.get():
            warning_key.config(text="Required")
            missing = True

        if missing:
            return

        # All valid, collect inputs
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
    tk.Button(root, text="Submit", command=submit).grid(row=6, column=0, columnspan=4, pady=20)

    # Auto-focus first field
    entry_file1.focus()

    # Bind Enter key: navigate or submit if last field
    def bind_enter_behavior(widget, is_last=False):
        if is_last:
            widget.bind("<Return>", lambda e: submit())
        else:
            widget.bind("<Return>", focus_next)

    fields = [
        entry_file1,
        entry_file2,
        entry_unique_key,
        entry_result_file_name,
        entry_file1_suffix,
        entry_file2_suffix  # last
    ]
    for i, widget in enumerate(fields):
        bind_enter_behavior(widget, is_last=(i == len(fields) - 1))

    root.mainloop()
    return result
