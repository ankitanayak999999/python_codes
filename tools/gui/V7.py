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

    # track slot row + its Browse button so we can realign/replace cleanly
    slot_rows = {}       # key -> row index
    slot_buttons = {}    # key -> Button widget

    # ---------- helpers ----------
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
            try:
                widget.destroy()
            except Exception:
                pass
        dynamic_widgets.clear()

        # clear any remembered slot buttons when removing a dynamic section
        for btn in list(slot_buttons.values()):
            try:
                btn.destroy()
            except Exception:
                pass
        slot_buttons.clear()

        keys_to_remove = [k for k in entries if 'dynamic__' in k]
        for k in keys_to_remove:
            del entries[k]

        keys_to_remove = [k for k in required_fields if 'dynamic__' in k]
        for k in keys_to_remove:
            del required_fields[k]

        for k in list(slot_rows.keys()):
            if k.startswith('dynamic__'):
                del slot_rows[k]

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

    # ---------- renderers ----------
    def render_file_input(row, label_text, required=False, key_prefix="static", filetypes=None):
        label = tk.Label(root, text=label_text, fg="red" if required else "black", anchor="w")
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

        entry = tk.Entry(root, width=85)
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")

        btn = tk.Button(root, text="Browse", command=lambda e=entry, ft=filetypes: browse_file(e, ft))
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

    # placeholder a radio can populate (file/dir). We still use a tiny frame only
    # to remember the row, but the Browse button is placed on the ROOT grid to align.
    def render_slot(row, label_text, required=False, key_prefix="static"):
        label = tk.Label(root, text=label_text, fg="red" if required else "black", anchor="w")
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

        frame = tk.Frame(root)                        # container for the entry only
        frame.grid(row=row, column=1, padx=5, pady=5, sticky="w")

        full_key = f"{key_prefix}__{label_text}"
        entries[full_key] = frame                     # will be replaced with Entry after populate
        slot_rows[full_key] = row
        if required:
            required_fields[full_key] = frame         # replaced with Entry after populate

        if key_prefix == "dynamic":
            dynamic_widgets.extend([label, frame])

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

    # ---------- builder ----------
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
                    if key_prefix == "dynamic":
                        dynamic_widgets.extend([lbl, entry])

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
                    if key_prefix == "dynamic":
                        dynamic_widgets.extend([lbl, combo])

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

                    # keep your focus chaining behavior
                    var.trace_add("write", lambda *args, idx=len(nav_widgets): nav_widgets[idx].focus_set())

                    # --- optional: radio -> slot control ---
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

                            # Destroy previous entry and button, if any
                            if slot_key in slot_buttons:
                                try:
                                    slot_buttons[slot_key].destroy()
                                except Exception:
                                    pass
                                del slot_buttons[slot_key]

                            # clear any widget inside the slot frame
                            row_idx = slot_rows.get(slot_key, row)
                            container = entries.get(slot_key)
                            if isinstance(container, tk.Entry):
                                # if we previously swapped value to Entry, grid row is taken from slot_rows
                                pass
                            else:
                                # container is the frame from render_slot
                                for child in container.winfo_children():
                                    child.destroy()

                            spec = (ctrl.get("map") or {}).get(selection, {})
                            spec_type = spec.get("type", "file")

                            # Entry goes inside the frame at column 0, button on ROOT at column 2
                            entry_parent = container if not isinstance(container, tk.Entry) else root

                            entry = tk.Entry(entry_parent, width=85)
                            if entry_parent is root:
                                entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
                            else:
                                entry.grid(row=0, column=0, padx=0, pady=0, sticky="w")

                            if spec_type == "file":
                                btn = tk.Button(root, text="Browse",
                                                command=lambda e=entry, ft=spec.get("filetypes"): browse_file(e, ft))
                            else:
                                btn = tk.Button(root, text="Browse", command=lambda e=entry: browse_dir(e))

                            btn.grid(row=row_idx, column=2, padx=5, pady=5, sticky="w")
                            slot_buttons[slot_key] = btn

                            # make submit/validation work on the Entry
                            entries[slot_key] = entry
                            required_fields[slot_key] = entry
                            nav_widgets.append(entry)

                        # first populate after slot exists; then live updates on toggle
                        root.after(0, lambda: _populate_slot(var.get()))
                        var.trace_add("write", lambda *_: _populate_slot(var.get()))

                    if key_prefix == "dynamic":
                        dynamic_widgets.extend([lbl, frame])

                widgets.append(lbl)

        row_counter[0] += len(field_list)
        return widgets

    # ---------- dynamic controller ----------
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

    # ---------- submit ----------
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

    # ---------- build form ----------
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
