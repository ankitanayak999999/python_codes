import tkinter as tk
from tkinter import filedialog, ttk

def get_user_inputs_conditional(controller_field, dynamic_field_map):
    result = {}

    root = tk.Tk()
    root.title("Dynamic Input Form")
    root.geometry("850x500")
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    widgets = []  # for dynamic cleanup
    field_vars = {}  # label → input var or entry
    current_row = 1

    def browse_file(entry):
        file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def clear_dynamic_fields():
        for widget in widgets:
            widget.destroy()
        widgets.clear()
        field_vars.clear()

    def build_dynamic_fields(selected_value):
        nonlocal current_row
        clear_dynamic_fields()
        dynamic_fields = dynamic_field_map.get(selected_value, [])
        for i, field in enumerate(dynamic_fields):
            label_text = field["label"] + (" *" if field.get("required") else "")
            tk.Label(root, text=label_text).grid(row=current_row+i, column=0, sticky="w", padx=5, pady=5)
            
            ftype = field.get("type", "text")

            if ftype == "text":
                entry = tk.Entry(root, width=70)
                entry.grid(row=current_row+i, column=1, padx=5, pady=5, sticky="w")
                field_vars[field["label"]] = entry
                widgets.append(entry)

            elif ftype == "file":
                frame = tk.Frame(root)
                frame.grid(row=current_row+i, column=1, padx=5, pady=5, sticky="w")
                entry = tk.Entry(frame, width=60)
                entry.grid(row=0, column=0, sticky="w")
                button = tk.Button(frame, text="Browse", command=lambda e=entry: browse_file(e))
                button.grid(row=0, column=1, padx=(5, 0), sticky="w")
                field_vars[field["label"]] = entry
                widgets.extend([frame, entry, button])

            elif ftype == "dropdown":
                var = tk.StringVar()
                combo = ttk.Combobox(root, textvariable=var, values=field["options"], width=67, state="readonly")
                combo.grid(row=current_row+i, column=1, padx=5, pady=5, sticky="w")
                field_vars[field["label"]] = combo
                widgets.append(combo)

            elif ftype == "radio":
                var = tk.StringVar()
                radio_frame = tk.Frame(root)
                radio_frame.grid(row=current_row+i, column=1, padx=5, pady=5, sticky="w")
                for option in field.get("options", []):
                    rb = tk.Radiobutton(radio_frame, text=option, variable=var, value=option)
                    rb.pack(side="left", padx=10)
                    widgets.append(rb)
                field_vars[field["label"]] = var
                widgets.append(radio_frame)

    def on_controller_change(event=None):
        selected = controller_var.get()
        build_dynamic_fields(selected)

    def submit():
        result["__controller__"] = controller_var.get()
        for label, widget in field_vars.items():
            if isinstance(widget, (tk.Entry, ttk.Combobox)):
                result[label] = widget.get()
            elif isinstance(widget, tk.StringVar):
                result[label] = widget.get()
        root.destroy()

    # Controller field
    tk.Label(root, text=controller_field["label"] + (" *" if controller_field.get("required") else ""), fg="red")\
        .grid(row=0, column=0, sticky="w", padx=5, pady=5)
    controller_var = tk.StringVar()
    controller_dropdown = ttk.Combobox(root, textvariable=controller_var, values=controller_field["options"], width=67, state="readonly")
    controller_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
    controller_dropdown.bind("<<ComboboxSelected>>", on_controller_change)

    # Submit button
    submit_btn = tk.Button(root, text="Submit", command=submit)
    submit_btn.grid(row=99, column=0, columnspan=2, pady=20)

    root.mainloop()
    return result


if __name__ == "__main__":
    controller_field = {
        "label": "Select Mode",
        "type": "dropdown",
        "options": ["value 1", "value 2", "value 3"],
        "required": True
    }

    dynamic_field_map = {
        "value 1": [
            {"label": "File A", "type": "file", "required": True}
        ],
        "value 2": [
            {"label": "File A", "type": "file"},
            {"label": "File B", "type": "file"}
        ],
        "value 3": [
            {"label": "File A", "type": "file"},
            {"label": "File B", "type": "file"},
            {"label": "Comment", "type": "text"}
        ]
    }

    output = get_user_inputs_conditional(controller_field, dynamic_field_map)
    print(output)
