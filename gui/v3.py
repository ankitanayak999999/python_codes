import tkinter as tk
from tkinter import filedialog, ttk, messagebox

def get_user_inputs(fields=None, controller_field=None, dynamic_field_map=None):
    result = {}
    root = tk.Tk()
    root.title("Input Form")
    root.geometry("900x500")
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    entries = {}
    required_fields = {}
    dynamic_widgets = []
    nav_widgets = []
    row_counter = [1]

    def browse_file(entry):
        file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def clear_dynamic_fields():
        for widget in dynamic_widgets:
            widget.destroy()
        dynamic_widgets.clear()
        keys_to_remove = [k for k in entries if '__dynamic__' in k]
        for k in keys_to_remove:
            del entries[k]
        keys_to_remove = [k for k in required_fields if '__dynamic__' in k]
        for k in keys_to_remove:
            del required_fields[k]

    def add_enter_navigation(widgets):
        for i, widget in enumerate(widgets):
            try:
                if i + 1 < len(widgets):
                    widget.bind("<Return>", lambda e, w=widgets[i + 1]: w.focus_set())
                else:
                    widget.bind("<Return>", lambda e: submit_btn.submit())
            except tk.TclError:
                pass

    def render_file_input(row, label_text, required=False, key_prefix="static"):
        label = tk.Label(root, text=label_text, fg="red" if required else "black", anchor="w")
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

        entry = tk.Entry(root, width=85)
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")

        btn = tk.Button(root, text="Browse", command=lambda e=entry: browse_file(e))
        btn.grid(row=row, column=2, padx=5, pady=5, sticky="w")

        full_key = f"{key_prefix}__{label_text}"
        entries[full_key] = entry
        if required:
            required_fields[full_key] = entry
        nav_widgets.append(entry)

        if key_prefix == "dynamic":
            dynamic_widgets.extend([label, entry, btn])

    def build_fields(field_list, start_row=0, key_prefix="static"):
        for i, field in enumerate(field_list):
            row = start_row + i
            label = field["label"]
            required = field.get("required", False)
            label_text = label + " *" if required else label
            fg_color = "red" if required else "black"
            ftype = field.get("type", "text")

            full_key = f"{key_prefix}__{label_text}"

            if ftype == "file":
                render_file_input(row, label_text, required, key_prefix)

            else:
                lbl = tk.Label(root, text=label_text, fg=fg_color, anchor="w")
                lbl.grid(row=row, column=0, padx=5, pady=5, sticky="w")

                if ftype == "text":
                    entry = tk.Entry(root, width=85)
                    entry.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="w")
                    entries[full_key] = entry
                    nav_widgets.append(entry)
                    if required:
                        required_fields[full_key] = entry

                elif ftype == "dropdown":
                    var = tk.StringVar()
                    combo = ttk.Combobox(root, textvariable=var, values=field.get("options", []), width=83, state="readonly")
                    combo.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="w")
                    entries[full_key] = combo
                    nav_widgets.append(combo)
                    if required:
                        required_fields[full_key] = combo

                elif ftype == "radio":
                    var = tk.StringVar()
                    frame = tk.Frame(root)
                    frame.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="w")
                    for opt in field.get("options", []):
                        tk.Radiobutton(frame, text=opt, variable=var, value=opt).pack(side="left", padx=0)
                    var.set(field.get("options", [])[0] if field.get("options") else "")
                    entries[full_key] = var
                    if required:
                        required_fields[full_key] = var

                if key_prefix == "dynamic":
                    dynamic_widgets.extend([lbl, entries[full_key]])

        row_counter[0] += len(field_list)

    def on_controller_change(event=None):
        selected = controller_var.get()
        clear_dynamic_fields()
        fields_to_add = dynamic_field_map.get(selected, [])
        build_fields(fields_to_add, start_row=row_counter[0], key_prefix="dynamic")
        nav_widgets.clear()
        for k, w in entries.items():
            if isinstance(w, (tk.Entry, ttk.Combobox)):
                nav_widgets.append(w)
        add_enter_navigation(nav_widgets)

    def submit():
        missing = []
        for key, widget in required_fields.items():
            if isinstance(widget, (tk.Entry, ttk.Combobox)):
                if not widget.get().strip():
                    missing.append(key.split("__", 1)[1])
            elif isinstance(widget, tk.StringVar):
                if not widget.get().strip():
                    missing.append(key.split("__", 1)[1])

        if missing:
            messagebox.showerror("Validation Error", "Missing required fields:\n" + "\n".join(missing))
            return

        if controller_field:
            result["__controller__"] = controller_var.get()

        for key, widget in entries.items():
            label = key.split("__", 1)[1]
            if isinstance(widget, (tk.Entry, ttk.Combobox)):
                result[label] = widget.get()
            elif isinstance(widget, tk.StringVar):
                result[label] = widget.get()

        root.destroy()

    # --- BUILD FORM ---
    if controller_field:
        tk.Label(root, text=controller_field["label"] + (" *" if controller_field.get("required") else ""), fg="red", anchor="w")\
            .grid(row=0, column=0, padx=5, pady=5, sticky="w")
        controller_var = tk.StringVar()
        dropdown = ttk.Combobox(root, textvariable=controller_var, values=controller_field["options"], width=83, state="readonly")
        dropdown.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="w")
        dropdown.bind("<<ComboboxSelected>>", on_controller_change)
        row_counter[0] = 1
        if controller_field.get("required"):
            required_fields["static__" + controller_field["label"]] = dropdown
        entries["static__" + controller_field["label"]] = dropdown
        nav_widgets.append(dropdown)

    elif fields:
        build_fields(fields, key_prefix="static")

    submit_btn = tk.Button(root, text="Submit", command=submit)
    submit_btn.grid(row=99, column=1, pady=20)
    add_enter_navigation(nav_widgets)
    root.mainloop()
    return result
