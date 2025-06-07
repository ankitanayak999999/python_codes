import pandas as pd
from tkinter import filedialog, Tk, Label, StringVar, Toplevel
import os
import time  # simulate long processing

def excel_to_csv(excel_file, csv_file, status_var):
    try:
        status_var.set("Reading Excel file...")
        df = pd.read_excel(excel_file, keep_default_na=False)

        status_var.set("Saving to CSV...")
        time.sleep(5)  # simulate long process
        df.to_csv(csv_file, index=False)

        status_var.set(f"Completed: {csv_file}")
    except Exception as e:
        status_var.set(f"Error: {e}")

def show_progress_window():
    progress = Toplevel()
    progress.title("Excel to CSV Tool")
    progress.geometry("400x100")
    status_var = StringVar(value="Starting...")

    Label(progress, textvariable=status_var, font=("Arial", 12)).pack(pady=20)
    return progress, status_var

def main():
    root = Tk()
    root.withdraw()

    excel_file = filedialog.askopenfilename(
        title="Select Excel File",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )

    if not excel_file:
        return

    # auto-generate CSV path in same folder
    path = os.path.dirname(excel_file)
    file_name = os.path.splitext(os.path.basename(excel_file))[0]
    csv_file = os.path.join(path, file_name + ".csv")

    progress_window, status_var = show_progress_window()
    root.after(100, lambda: excel_to_csv(excel_file, csv_file, status_var))
    root.mainloop()

if __name__ == "__main__":
    main()
