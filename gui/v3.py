import tkinter as tk
from tkinter import filedialog, ttk

def get_user_inputs_dynamic_dynamic_fields(controller_field, dynamic_field_map):
    result = {}
    root = tk.Tk()
    root.title("Dynamic Input Form")
    root.geometry("850x500")
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    entries = {}
    warnings = []
    dynamic_widgets = []

    def focus_next(event):
        event.widget.tk_focusNext().focus()
        return "break"

    def browse_file(entry):
        file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def clear_dynamic_fields():
        for widget in dynamic_widgets:
            widget.destroy()
        dynamic_widgets.clear()

    def build_dynamic_fields(selected_value, start_row=1):
        clear_dynamic_fields()
        dynamic_fields = dynamic_field_map.get(selected_value, [])
        for idx, field in enumerate(dynamic_fields):
            label_text = f"{field['label']} *" if field.get("required") else field["label"]
            fg_color = "red" if field.get("required") else "black"
            tk.Label(root, text=label_text, fg=fg_color)\
                .grid(row=start_row + idx, column=0, padx=5, pady=5, sticky="w")

            ftype = field.get("type", "text")

            if ftype == "text":
                entry = tk.Entry(root, width=70)
                entry.grid(row=start_row + idx, column=1, padx=5, pady=5, sticky="w")
                entries[field["label"]] = entry
                dynamic_widgets.extend([entry])

            elif ftype == "file":
                frame = tk.Frame(root)
                frame.grid(row=start_row + idx, column=1, padx=5, pady=5, sticky="w")

                entry = tk.Entry(frame, width=60)
                entry.grid(row=0, column=0, sticky="w")

                btn = tk.Button(frame, text="Browse", command=lambda e=entry: browse_file(e))
                btn.grid(row=0, column=1, padx=(5, 0), sticky="w")

                entries[field["label"]] = entry
                dynamic_widgets.extend([frame, entry, btn])

            elif ftype == "radio":
                var = tk.StringVar()
                radio_frame = tk.Frame(root)
                radio_frame.grid(row=start_row + idx, column=1, sticky="w", padx=5, pady=5)
                for opt in field.get("options", []):
                    rb = tk.Radiobutton(radio_frame, text=opt, variable=var, value=opt)
                    rb.pack(side="left", padx=10)
                if field.get("options"):
                    var.set(field["options"][0])
                entries[field["label"]] = var
                dynamic_widgets.extend([radio_frame])

            elif ftype == "dropdown":
                var = tk.StringVar()
                combo = ttk.Combobox(root, textvariable=var, values=field.get("options", []), width=67, state="readonly")
                combo.grid(row=start_row + idx, column=1, padx=5, pady=5, sticky="w")
                entries[field["label"]] = combo
                dynamic_widgets.append(combo)

    def on_controller_change(event=None):
        selected = controller_var.get()
        build_dynamic_fields(selected)

    def submit():
        result["__controller__"] = controller_var.get()
        for label, widget in entries.items():
            if isinstance(widget, (tk.Entry, ttk.Combobox)):
                result[label] = widget.get()
            elif isinstance(widget, tk.StringVar):
                result[label] = widget.get()
        root.destroy()

    # Controller dropdown
    tk.Label(root, text=controller_field["label"] + (" *" if controller_field.get("required") else ""), fg="red")\
        .grid(row=0, column=0, sticky="w", padx=5, pady=5)

    controller_var = tk.StringVar()
    controller_dropdown = ttk.Combobox(root, textvariable=controller_var,
                                       values=controller_field["options"],
                                       width=67, state="readonly")
    controller_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
    controller_dropdown.bind("<<ComboboxSelected>>", on_controller_change)

    # Submit button
    submit_btn = tk.Button(root, text="Submit", command=submit)
    submit_btn.grid(row=99, column=0, columnspan=2, pady=20)

    root.mainloop()
    return result


if __name__ == "__main__":
    controller_field = {
        "label": "Input Type",
        "type": "dropdown",
        "options": ["value 1", "value 2", "value 3"],
        "required": True
    }

    dynamic_field_map = {
        "value 1": [
            {"label": "Upload File A", "type": "file", "required": True}
        ],
        "value 2": [
            {"label": "Upload File A", "type": "file"},
            {"label": "Upload File B", "type": "file"}
        ],
        "value 3": [
            {"label": "Upload File A", "type": "file"},
            {"label": "Upload File B", "type": "file"},
            {"label": "Enter Comment", "type": "text"}
        ]
    }

    inputs = get_user_inputs_dynamic_dynamic_fields(controller_field, dynamic_field_map)
    print(inputs)
