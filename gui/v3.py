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

    def build_fields(field_list, start_row=0):
        for i, field in enumerate(field_list):
            row = start_row + i
            label_text = f"{field['label']} *" if field.get("required") else field["label"]
            fg_color = "red" if field.get("required") else "black"

            tk.Label(root, text=label_text, fg=fg_color)\
                .grid(row=row, column=0, padx=5, pady=5, sticky="w")

            ftype = field.get("type", "text")

            if ftype == "text":
                entry = tk.Entry(root, width=70)
                entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                entries[field["label"]] = entry

            elif ftype == "file":
                entry = tk.Entry(root, width=60)
                entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                btn = tk.Button(root, text="Browse", command=lambda e=entry: browse_file(e))
                btn.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                entries[field["label"]] = entry

            elif ftype == "dropdown":
                var = tk.StringVar()
                combo = ttk.Combobox(root, textvariable=var, values=field.get("options", []), width=67, state="readonly")
                combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
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
                entries[field["label"]] = var

    def build_dynamic_fields(selection):
        clear_dynamic_fields()
        fields = dynamic_field_map.get(selection, [])
        for i, field in enumerate(fields):
            row = row_counter[0] + i
            label_text = f"{field['label']} *" if field.get("required") else field["label"]
            fg_color = "red" if field.get("required") else "black"

            label = tk.Label(root, text=label_text, fg=fg_color)
            label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
            dynamic_widgets.append(label)

            ftype = field.get("type", "text")

            if ftype == "text":
                entry = tk.Entry(root, width=70)
                entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                dynamic_widgets.append(entry)
                entries[field["label"]] = entry

            elif ftype == "file":
                entry = tk.Entry(root, width=60)
                entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
                btn = tk.Button(root, text="Browse", command=lambda e=entry: browse_file(e))
                btn.grid(row=row, column=2, padx=5, pady=5, sticky="w")
                dynamic_widgets.extend([entry, btn])
                entries[field["label"]] = entry

            elif ftype == "dropdown":
                var = tk.StringVar()
                combo = ttk.Combobox(root, textvariable=var, values=field.get("options", []), width=67, state="readonly")
                combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
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
                dynamic_widgets.append(radio_frame)
                entries[field["label"]] = var

        row_counter[0] += len(fields)

    def on_controller_change(event=None):
        selected = controller_var.get()
        build_dynamic_fields(selected)

    def submit():
        if controller_field:
            result["__controller__"] = controller_var.get()
        for label, widget in entries.items():
            if isinstance(widget, (tk.Entry, ttk.Combobox)):
                result[label] = widget.get()
            elif isinstance(widget, tk.StringVar):
                result[label] = widget.get()
        root.destroy()

    # --- Build UI ---
    if controller_field:
        # Row 0: Controller dropdown
        tk.Label(root, text=controller_field["label"] + (" *" if controller_field.get("required") else ""), fg="red")\
            .grid(row=0, column=0, padx=5, pady=5, sticky="w")

        controller_var = tk.StringVar()
        dropdown = ttk.Combobox(root, textvariable=controller_var, values=controller_field["options"], width=67, state="readonly")
        dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        dropdown.bind("<<ComboboxSelected>>", on_controller_change)
        row_counter[0] = 1  # next row for dynamic fields
    else:
        build_fields(fields or [], start_row=0)

    # Submit Button (bottom row)
    tk.Button(root, text="Submit", command=submit)\
        .grid(row=99, column=0, columnspan=3, pady=20)

    root.mainloop()
    return result
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
