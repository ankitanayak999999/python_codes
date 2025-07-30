import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def get_user_inputs_dynamic(fields):
    result = []

    root = tk.Tk()
    root.title("Provide Inputs")
    root.geometry(f"850x{130 + 60 * len(fields)}")

    # Force GUI to front
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
        result = []

        for i, entry in enumerate(entries):
            if isinstance(entry, (tk.Entry, ttk.Combobox)):
                val = entry.get()
            elif isinstance(entry, tk.StringVar):
                val = entry.get()
            else:
                val = ""

            warnings[i].config(text="")

            if fields[i].get("required") and not val.strip():
                warnings[i].config(text="Required")
                valid = False

            result.append(val)

        if not valid:
            result.clear()
            return

        root.destroy()

    def bind_enter_behavior(widget, is_last=False):
        if is_last:
            widget.bind("<Return>", lambda e: submit())
        else:
            widget.bind("<Return>", focus_next)

    for i, field in enumerate(fields):
        label_text = field.get("label", f"Field {i+1}")
        ftype = field.get("type", "text")
        required = field.get("required", False)
        options = field.get("options", [])

        # Label
        tk.Label(root, text=label_text + (" *" if required else ""), fg="red" if required else "black")\
            .grid(row=i, column=0, sticky="w", padx=5, pady=5)

        # Validation warning
        warning = tk.Label(root, text="", fg="red")
        warning.grid(row=i, column=2, sticky="w")
        warnings.append(warning)

        if ftype == "text":
            entry = tk.Entry(root, width=70)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            entries.append(entry)
            bind_enter_behavior(entry, is_last=(i == len(fields) - 1))

        elif ftype == "file":
            frame = tk.Frame(root)
            frame.grid(row=i, column=1, padx=5, pady=5, sticky="w")

            entry = tk.Entry(frame, width=60)
            entry.grid(row=0, column=0, sticky="w")

            # Prevent typing — soft read-only
            def block_typing(event):
                return "break"
            entry.bind("<Key>", block_typing)
            entry.bind("<Control-v>", block_typing)
            entry.bind("<Button-3>", block_typing)

            browse_btn = tk.Button(frame, text="Browse", command=lambda e=entry: browse_file(e))
            browse_btn.grid(row=0, column=1, padx=(5, 0), sticky="w")

            entries.append(entry)
            bind_enter_behavior(entry, is_last=(i == len(fields) - 1))

        elif ftype == "dropdown":
            selected = tk.StringVar()
            entry = ttk.Combobox(root, textvariable=selected, values=options, width=67, state="readonly")
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            entries.append(entry)
            bind_enter_behavior(entry, is_last=(i == len(fields) - 1))

        elif ftype == "radio":
            selected = tk.StringVar()
            entries.append(selected)

            radio_frame = tk.Frame(root)
            radio_frame.grid(row=i, column=1, sticky="w", padx=5, pady=5)

            for j, opt in enumerate(options):
                rb = tk.Radiobutton(radio_frame, text=opt, variable=selected, value=opt)
                rb.pack(side="left", padx=10)

            selected.set(options[0] if options else "")

    # Submit button
    tk.Button(root, text="Submit", command=submit)\
        .grid(row=len(fields), column=0, columnspan=4, pady=20)

    root.mainloop()
    return result
