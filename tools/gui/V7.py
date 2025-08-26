# gui_engine_v7.py  (v6-compatible + dir + filetypes + slot-control)
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

    # -------- helpers (v6-compatible) --------
    def browse_file(entry, filetypes=None):
        ftypes = filetypes or [("All files", "*.*")]
        file_path = filedialog.askopenfilename(filetypes=ftypes)
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def browse_dir(entry):
        folder_path = filedialog.askdirectory(title="Select a folder")
        if folder_path:
            entry.delete(0, tk.END)
            entry.insert(0, folder_path)

    def clear_dynamic_fields():
        for widget in dynamic_widgets:
            widget.destroy()
        dynamic_widgets.clear()

        keys_to_remove = [k for k in entries if 'dynamic__' in k]
        for k in keys_to_remove:
            del entries[k]

        keys_to_remove = [k for k in required_fields if 'dynamic__' in k]
        for k in keys_to_remove:
            del required_fields[k]

    def add_enter_navigation(widgets):
        navigable = [w for w in widgets if isinstance(w, (tk.Entry, ttk.Combobox))]
        for i, widget in enumerate(navigable):
            try:
                if i + 1 < len(navigable):
                    widget.bind("<Return>", lambda e, w=navigable[i + 1]: w.focus_set())
                else:
                    widget.bind("<Return>", lambda e: submit_btn.invoke())
            except tk.TclError:
                pass

    # -------- renderers (v6 style) --------
    def render_file_input(row, label_text, required=False, key_prefix="static", filetypes=None):
        label = tk.Label(root, text=label_text, fg="red" if required else "black", anchor="w")
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

        entry = tk.Entry(root, width=85)
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")

        btn = tk.Button(root, text="Browse",
                        command=lambda e=entry, ft=filetypes: browse_file(e, ft))
        btn.grid(row=row, column=2, padx=5, pady=5, sticky="w")

        full_key = f"{key_prefix}__{label_text}"
        entries[full_key] = entry
        if required:
            required_fields[full_key] = entry

        if key_prefix == "dynamic":
            dynamic_widgets.extend([label, entry, btn])

        nav_widgets.append(entry)
        return [entry, btn]

    def render_dir_input(row, label_text, required=False, key_prefix="static"):
        label = tk.Label(root, text=label_text, fg="red" if required else "black", anchor="w")
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

        entry = tk.Entry(root, width=85)
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")

        btn = tk.Button(root, text="Browse", command=lambda e=entry: browse_dir(e))
        btn.grid(row=row, column=2, padx=5, pady=5, sticky="w")

        full_key = f"{key_prefix}__{label_text}"
        entries[full_key] = entry
        if required:
            required_fields[full_key] = entry

        if key_prefix == "dynamic":
            dynamic_widgets.extend([label, entry, btn])

        nav_widgets.append(entry)
        return [entry, btn]

    # NEW: placeholder slot the radio can populate
    def render_slot(row, label_text, required=False, key_prefix="static"):
        label = tk.Label(root, text=label_text, fg="red" if required else "black", anchor="w")
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

        frame = tk.Frame(root)
        frame.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        full_key = f"{key_prefix}__{label_text}"
        entries[full_key] = frame
        if required:
            required_fields[full_key] = frame

        if key_prefix == "dynamic":
            dynamic_widgets.extend([label, frame])

        # no nav widget yet (added when populated)
        return [frame]

    def bind_dropdown_focus(combo_widget):
        def on_select(event):
            try:
                idx = nav_widgets.index(combo_widget)
                if idx + 1 < len(nav_widgets):
                    nav_widgets[idx + 1].focus_set()
            except Exception:
                pass
        combo_widget.bind("<<ComboboxSelected>>", on_select)

    # -------- builder (v6-compatible shape) --------
    def build_fields(field_list, start_row=0, key_prefix="static"):
        widgets = []
        for i, field in enumerate(field_list):
            row = start_row + i
            label = field["label"]
            required = field.get("required", False)
            label_text = label + " *" if required else label
            fg_color = "red" if required else "black"
            ftype = field.get("type", "text")

            full_key = f"{key_prefix}__{label_text}"

            if ftype == "file":
                widgets.extend(
                    render_file_input(row, label_text, required, key_prefix, filetypes=field.get("filetypes"))
                )

            elif ftype == "dir":
                widgets.extend(render_dir_input(row, label_text, required, key_prefix))

            elif ftype == "slot":
                widgets.extend(render_slot(row, label_text, required, key_prefix))

            else:
                lbl = tk.Label(root, text=label_text, fg=fg_color, anchor="w")
                lbl.grid(row=row, column=0, padx=5, pady=5, sticky="w")

                if ftype == "text":
                    entry = tk.Entry(root, width=85)
                    entry.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="w")
                    entries[full_key] = entry
                    nav_widgets.append(entry)
                    widgets.append(entry)
                    if required:
                        required_fields[full_key] = entry

                elif ftype == "dropdown":
                    var = tk.StringVar()
                    combo = ttk.Combobox(root, textvariable=var,
                                         values=field.get("options", []),
                                         width=82, state="readonly")
                    combo.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="w")
                    entries[full_key] = combo
                    nav_widgets.append(combo)
                    widgets.append(combo)
                    if required:
                        required_fields[full_key] = combo
                    bind_dropdown_focus(combo)

                elif ftype == "radio":
                    var = tk.StringVar()
                    frame = tk.Frame(root)
                    frame.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="w")
                    for opt in field.get("options", []):
                        tk.Radiobutton(frame, text=opt, variable=var, value=opt).pack(side="left", padx=0)
                    var.set(field.get("options", [])[0] if field.get("options") else "")
                    entries[full_key] = var
                    widgets.append(frame)
                    if required:
                        required_fields[full_key] = var

                    # NEW: optional radio -> slot controller
                    ctrl = field.get("controls")
                    if ctrl:
                        raw_target = ctrl.get("target", "")
                        target_base = raw_target.replace("*", "").strip()

                        def _find_slot_key():
                            for k in entries.keys():
                                suffix = k.split("__", 1)[1]
                                if suffix.replace("*", "").strip() == target_base:
                                    return k
                            return None

                        def _populate_slot(selection):
                            slot_key = _find_slot_key()
                            if not slot_key:
                                return
                            container = entries.get(slot_key)
                            # if slot already has an Entry, get its parent frame
                            if isinstance(container, tk.Entry):
                                container = container.master
                            # clear
                            for child in container.winfo_children():
                                child.destroy()

                            spec = (ctrl.get("map") or {}).get(selection, {})
                            spec_type = spec.get("type", "file")

                            if spec_type == "file":
                                entry = tk.Entry(container, width=85)
                                entry.grid(row=0, column=0, padx=0, pady=0, sticky="w")
                                btn = tk.Button(container, text="Browse",
                                                command=lambda e=entry, ft=spec.get("filetypes"): browse_file(e, ft))
                                btn.grid(row=0, column=1, padx=5, pady=0, sticky="w")
                                entries[slot_key] = entry
                                required_fields[slot_key] = entry
                                nav_widgets.append(entry)

                            elif spec_type == "dir":
                                entry = tk.Entry(container, width=85)
                                entry.grid(row=0, column=0, padx=0, pady=0, sticky="w")
                                btn = tk.Button(container, text="Browse", command=lambda e=entry: browse_dir(e))
                                btn.grid(row=0, column=1, padx=5, pady=0, sticky="w")
                                entries[slot_key] = entry
                                required_fields[slot_key] = entry
                                nav_widgets.append(entry)

                        # initial & on change
                        _populate_slot(var.get())
                        var.trace_add("write", lambda *_: _populate_slot(var.get()))

                    # keep your existing focus chaining semantics
                    var.trace_add("write", lambda *args, idx=len(nav_widgets): nav_widgets[idx].focus_set())

                widgets.append(lbl)

                if key_prefix == "dynamic":
                    # for simple types, ensure we can clear later
                    if full_key in entries:
                        dynamic_widgets.extend([lbl])

        row_counter[0] += len(field_list)
        return widgets

    # -------- dynamic controller --------
    def on_controller_change(event=None):
        selected = controller_var.get()
        clear_dynamic_fields()
        fields_to_add = dynamic_field_map.get(selected, [])
        new_widgets = build_fields(fields_to_add, start_row=row_counter[0], key_prefix="dynamic")

        for w in new_widgets:
            if isinstance(w, (tk.Entry, ttk.Combobox)):
                w.focus_set()
                break

        add_enter_navigation(new_widgets)

    # -------- submit & collect (v6 style) --------
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

        for key, widget in entries.items():
            raw_label = key.split("__", 1)[1]
            label = raw_label.replace("*", "").strip()
            if isinstance(widget, (tk.Entry, ttk.Combobox)):
                result[label] = widget.get()
            elif isinstance(widget, tk.StringVar):
                result[label] = widget.get()

        root.destroy()

    # -------- build form (v6 layout preserved) --------
    if controller_field:
        tk.Label(root, text=controller_field["label"] + (" *" if controller_field.get("required") else ""),
                 fg="red" if controller_field.get("required") else "", anchor="w")\
            .grid(row=0, column=0, padx=5, pady=5, sticky="w")

        controller_var = tk.StringVar()
        dropdown = ttk.Combobox(root, textvariable=controller_var,
                                values=controller_field["options"], width=82, state="readonly")
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
    if nav_widgets:
        nav_widgets[0].focus_set()

    root.mainloop()
    return result

if __name__ == "__main__":
    print("gui engine started........")
