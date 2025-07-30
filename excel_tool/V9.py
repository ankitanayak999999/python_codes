import tkinter as tk
from tkinter import filedialog

def get_user_inputs_dynamic(fields):
    """
    Dynamically creates a GUI form based on the given fields.

    Parameters:
        fields: list of dicts, each with:
            - "label": str, the label text
            - "required": bool, whether field is mandatory
            - "type": "file" or "text" (default is "text")

    Returns:
        List of user inputs (in the same order as fields)
    """
    if not fields:
        raise ValueError("At least one field is required.")

    result = None
    root = tk.Tk()
    root.title("Provide Inputs")
    root.geometry(f"850x{130 + 40 * len(fields)}")  # Auto-size window

    entries = []
    warnings = []

    def focus_next(event):
        event.widget.tk_focusNext().focus()
        return "break"

    def browse_file(entry):
        file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def submit():
        nonlocal result
        valid = True

        for i, entry in enumerate(entries):
            warnings[i].config(text="")
            if fields[i].get("required") and not entry.get():
                warnings[i].config(text="Required")
                valid = False

        if not valid:
            return

        result = [entry.get() for entry in entries]
        root.destroy()

    for i, field in enumerate(fields):
        label_text = f"{field['label']} {'*' if field.get('required') else ''}"
        fg_color = "red" if field.get("required") else "black"

        tk.Label(root, text=label_text, fg=fg_color).grid(row=i, column=0, padx=5, pady=5, sticky="w")

        entry = tk.Entry(root, width=70)
        entry.grid(row=i, column=1, padx=5)

        if field.get("type") == "file":
            tk.Button(root, text="Browse", command=lambda e=entry: browse_file(e)).grid(row=i, column=2, padx=5)

        warning = tk.Label(root, text="", fg="red")
        warning.grid(row=i, column=3, sticky="w")

        entry.bind("<Return>", focus_next)
        entries.append(entry)
        warnings.append(warning)

    entries[0].focus()
    entries[-1].bind("<Return>", lambda e: submit())
    tk.Button(root, text="Submit", command=submit).grid(row=len(fields) + 1, column=0, columnspan=3, pady=20)

    root.mainloop()
    return result


from gui_engine import get_user_inputs_dynamic

fields = [
    {"label": "Select File 1", "required": True, "type": "file"},
    {"label": "Select File 2", "required": True, "type": "file"},
    {"label": "Unique ID Column", "required": True, "type": "text"},
    {"label": "Result File Name", "required": False, "type": "text"},
    {"label": "File 1 Suffix", "required": False, "type": "text"},
    {"label": "File 2 Suffix", "required": False, "type": "text"}
]

inputs = get_user_inputs_dynamic(fields)
print(inputs)



fields = [
    {"label": "Upload Dataset", "required": True, "type": "file"},
    {"label": "Reviewer Name", "required": True, "type": "text"},
    {"label": "Remarks", "required": False, "type": "text"}
]

inputs = get_user_inputs_dynamic(fields)
print(inputs)



