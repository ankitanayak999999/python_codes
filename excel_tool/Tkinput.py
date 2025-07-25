import tkinter as tk

def get_inputs():
    val1 = entry1.get()
    val2 = entry2.get()
    val3 = entry3.get()
    print("Input 1:", val1)
    print("Input 2:", val2)
    print("Input 3:", val3)
    root.destroy()  # Close window after submission

# Create window
root = tk.Tk()
root.title("Enter 3 Inputs")

# Labels and entry fields
tk.Label(root, text="Enter Input 1:").pack()
entry1 = tk.Entry(root, width=40)
entry1.pack()

tk.Label(root, text="Enter Input 2:").pack()
entry2 = tk.Entry(root, width=40)
entry2.pack()

tk.Label(root, text="Enter Input 3:").pack()
entry3 = tk.Entry(root, width=40)
entry3.pack()

# Submit button
tk.Button(root, text="Submit", command=get_inputs).pack()

root.mainloop()
