import tkinter as tk
from tkinter import filedialog, ttk

def get_user_inputs(fields=None, controller_field=None, dynamic_field_map=None):
    result = {}
    root = tk.Tk()
    root.title("Dynamic Input Form")
    root.geometry("850x500")
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    entries = {}
    dynamic_widgets = []
    row_counter = [1]  # to track current row for dynamic fields

    def browse_file(entry):
        file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def clear_dynamic_fields():
        for widget in dynamic_widgets:
            widget.destroy()
        dynamic_widgets.clear()

    def render_file_input(row, label_text, fg_color, target_dict, parent="static"):
        tk.Label(root, text=label_text, fg=fg_color).grid(row=row, column=0, padx=5, pady=5, sticky="w")
        file_frame = tk.Frame(root)
        file_frame.grid(row=row, column=1, padx=5, pady=5, sticky="w")

        entry = tk.Entry(file_frame, width=60)
        entry.grid(row=0, column=0, sticky="w")

        btn = tk.Button(file_frame, text="Browse", command=lambda e=entry: browse_file(e))
        btn.grid(row=0, column=1, padx=(5, 0), sticky="w")

        if parent == "dynamic":
            dynamic_widgets.extend([file_frame, entry, btn])
        target_dict[label_text.rstrip(" *")] = entry

    def build_fields(field_list, start_row=0, parent="static"):
        for i, field in enumerate(field_list):
            row = start_row + i
            label_text = f"{field['label']} *" if field.get("required") else field["label"]
            fg_color = "red" if field.get("required") else "black"

            ftype = field.get("type", "text")

            if ftype == "file":
                render_file_input(row, label_text, fg_color, entries, parent)

            else:
                tk.Label(root, text=label_text, fg=fg_color).grid(row=row, column=0, padx=5, pady=5, sticky="w")

                if ftype == "text":
                    entry = tk.Entry(root, width=70)
                    entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                    if parent == "dynamic":
                        dynamic_widgets.append(entry)
                    entries[field["label"]] = entry

                elif ftype == "dropdown":
                    var = tk.StringVar()
                    combo = ttk.Combobox(root, textvariable=var, values=field.get("options", []), width=67, state="readonly")
                    combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                    if parent == "dynamic":
                        dynamic_widgets.append(combo)
                    entries[field["label"]] = combo

                elif ftype == "radio":
                    var = tk.StringVar()
                    radio_frame = tk.Frame(root)
                    radio_frame.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                    for opt in field.get("options", []):
                        rb = tk.Radiobutton(radio_frame, text=opt, variable=var, value=opt)
                        rb.pack(side="left", padx=10)
                    if field.get("options"):
                        var.set(field["options"][0])
                    if parent == "dynamic":
                        dynamic_widgets.append(radio_frame)
                    entries[field["label"]] = var

        if parent == "dynamic":
            row_counter[0] += len(field_list)

    def on_controller_change(event=None):
        selected = controller_var.get()
        clear_dynamic_fields()
        if dynamic_field_map:
            build_fields(dynamic_field_map.get(selected, []), start_row=row_counter[0], parent="dynamic")

    def submit():
        if controller_field:
            result["__controller__"] = controller_var.get()
        for label, widget in entries.items():
            if isinstance(widget, (tk.Entry, ttk.Combobox)):
                result[label] = widget.get()
            elif isinstance(widget, tk.StringVar):
                result[label] = widget.get()
        root.destroy()

    # --- BUILD UI ---
    if controller_field:
        tk.Label(root, text=controller_field["label"] + (" *" if controller_field.get("required") else ""), fg="red")\
            .grid(row=0, column=0, padx=5, pady=5, sticky="w")

        controller_var = tk.StringVar()
        dropdown = ttk.Combobox(root, textvariable=controller_var, values=controller_field["options"], width=67, state="readonly")
        dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        dropdown.bind("<<ComboboxSelected>>", on_controller_change)
        row_counter[0] = 1  # Start row for dynamic fields
    else:
        build_fields(fields or [], start_row=0, parent="static")

    tk.Button(root, text="Submit", command=submit).grid(row=99, column=0, columnspan=3, pady=20)
    root.mainloop()
    return result
-------------------------------------------------
fields = [
    {"label": "Username", "type": "text", "required": True},
    {"label": "Upload Report", "type": "file"},
    {"label": "Format", "type": "radio", "options": ["PDF", "Excel", "HTML"]},
    {"label": "Department", "type": "dropdown", "options": ["IT", "HR", "Sales"]}
]

inputs = get_user_inputs(fields=fields)
print(inputs)

controller_field = {
    "label": "Input Type",
    "type": "dropdown",
    "options": ["Simple", "Advanced"]
}

dynamic_field_map = {
    "Simple": [{"label": "File A", "type": "file"}],
    "Advanced": [{"label": "File A", "type": "file"}, {"label": "Comment", "type": "text"}]
}

inputs = get_user_inputs(controller_field=controller_field, dynamic_field_map=dynamic_field_map)
print(inputs)
