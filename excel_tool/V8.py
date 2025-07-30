import tkinter as tk
from tkinter import messagebox

def get_inputs_with_requirements(fields):
    """
    Builds a Tkinter form based on a list of field definitions.
    Each field must be a dict with:
        - "label": str (field label)
        - "required": bool (whether it's mandatory)

    Returns:
        List of strings (user inputs, in order)
    """
    result = None
    root = tk.Tk()
    root.title("Provide Inputs")
    root.geometry("700x{}".format(130 + 40 * len(fields)))

    entries = []
    warnings = []

    def focus_next(event):
        event.widget.tk_focusNext().focus()
        return "break"

    def submit():
        nonlocal result
        valid = True

        for i, entry in enumerate(entries):
            warnings[i].config(text="")
            if fields[i].get("required", False) and not entry.get():
                warnings[i].config(text="Required")
                valid = False

        if not valid:
            return

        result = [entry.get() for entry in entries]
        root.destroy()

    # Build the form
    for i, field in enumerate(fields):
        label_text = f"{field['label']} {'*' if field.get('required') else ''}"
        fg_color = "red" if field.get("required") else "black"

        tk.Label(root, text=label_text, fg=fg_color).grid(row=i, column=0, padx=5, pady=5, sticky="w")

        entry = tk.Entry(root, width=60)
        entry.grid(row=i, column=1, padx=5)

        warning = tk.Label(root, text="", fg="red")
        warning.grid(row=i, column=2, sticky="w")

        entry.bind("<Return>", focus_next)
        entries.append(entry)
        warnings.append(warning)

    # Set focus and submit bindings
    entries[0].focus()
    entries[-1].bind("<Return>", lambda e: submit())

    tk.Button(root, text="Submit", command=submit).grid(
        row=len(fields) + 1, column=0, columnspan=3, pady=20
    )

    root.mainloop()
    return result



from gui_engine import get_inputs_with_requirements

fields = [
    {"label": "Enter File 1", "required": True},
    {"label": "Enter File 2", "required": True},
    {"label": "Join Key", "required": True},
    {"label": "Notes (Optional)", "required": False}
]

inputs = get_inputs_with_requirements(fields)

if inputs:
    print("User Inputs:", inputs)
    # use inputs[0], inputs[1], etc. in your toolto C
