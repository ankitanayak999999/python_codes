import tkinter as tk
from tkinter import ttk, messagebox

def show_selection():
    """Display the selected dropdown value."""
    selected_value = dropdown_var.get()
    messagebox.showinfo("Selection", f"You selected: {selected_value}")

# Create main window
root = tk.Tk()
root.title("Dropdown Example")
root.geometry("300x150")  # Optional: Set window size

# Dropdown label
tk.Label(root, text="Select an option:").pack(anchor=tk.W, padx=10, pady=5)

# Dropdown (Combobox)
dropdown_var = tk.StringVar()
dropdown = ttk.Combobox(root, textvariable=dropdown_var)
dropdown['values'] = ("Choice A", "Choice B", "Choice C")  # Dropdown values
dropdown.current(0)  # Set default selected value
dropdown.pack(anchor=tk.W, padx=20)

# Submit button
tk.Button(root, text="Submit", command=show_selection).pack(pady=10)

# Run the GUI loop
root.mainloop()
