import tkinter as tk
from tkinter import filedialog, ttk

def get_user_inputs_dynamic(fields):
    if not fields:
        raise ValueError("At least one field is required.")

    result = []
    root = tk.Tk()
    root.title("Provide Inputs")
    root.geometry(f"850x{130 + 60 * len(fields)}")

    # Bring window to front
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

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
        result.clear()

        for i, entry in enumerate(entries):
            val = entry.get() if isinstance(entry, (tk.Entry, ttk.Combobox, tk.StringVar)) else ""
            warnings[i].config(text="")
            if fields[i].get("required") and not val.strip():
                warnings[i].config(text="Required")
                valid = False
            result.append(val)

        if not valid:
            result.clear()
            return

        root.destroy()

    for i, field in enumerate(fields):
        ftype = field.get("type", "text")
        label_text = f"{field['label']} {'*' if field.get('required') else ''}"
        fg_color = "red" if field.get("required") else "black"

        tk.Label(root, text=label_text, fg=fg_color).grid(row=i, column=0, padx=5, pady=5, sticky="nw")

        if ftype == "dropdown":
            combo = ttk.Combobox(root, values=field.get("options", []), width=67, state="readonly")
            combo.grid(row=i, column=1, padx=5, sticky="w")
            combo.bind("<Return>", focus_next)
            entries.append(combo)

        elif ftype == "radio":
            var = tk.StringVar()
            options = field.get("options", [])
            if options:
                var.set(options[0])  # Default selection

            radio_frame = tk.Frame(root)
            for option in options:
                tk.Radiobutton(radio_frame, text=option, variable=var, value=option).pack(side="left", padx=5)
            radio_frame.grid(row=i, column=1, columnspan=3, sticky="w")

            entries.append(var)

        else:
            entry = tk.Entry(root, width=70)
            entry.grid(row=i, column=1, padx=5)
            if ftype == "file":
                tk.Button(root, text="Browse", command=lambda e=entry: browse_file(e)).grid(row=i, column=2, padx=5)
            entry.bind("<Return>", focus_next)
            entries.append(entry)

        warning = tk.Label(root, text="", fg="red")
        warning.grid(row=i, column=5, sticky="w")
        warnings.append(warning)

    entries[0].focus()
    if hasattr(entries[-1], "bind"):
        entries[-1].bind("<Return>", lambda e: submit())

    tk.Button(root, text="Submit", command=submit).grid(row=len(fields) + 1, column=0, columnspan=4, pady=20)
    root.mainloop()

    return result
