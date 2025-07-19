import pandas as pd
import os
import sqlparse
from tkinter import Tk, filedialog
import datetime

def gui_selection(selection_type, comment_str):
    if selection_type == 'file':
        return filedialog.askopenfilename(title=comment_str, filetypes=[("csv files", "*.csv"), ("Excel files", "*.xlsx")])
    return None

# Step 1: Split SQL definitions into .sql files
def split_sql_files(sql_csv_file, path):
    df1 = pd.read_csv(sql_csv_file, keep_default_na=False, low_memory=False)
    df1 = df1[['OBJECT_NAME', 'DDL']]

    view_sql_path = os.path.join(path, "view_sqls")
    os.makedirs(view_sql_path, exist_ok=True)

    for _, data in df1.iterrows():
        view_name = data['OBJECT_NAME']
        ddl = data['DDL']

        # Format SQL
        formatted_sql = sqlparse.format(ddl, reindent=True, keyword_case='upper')

        # Safe file name
        safe_view_name = view_name.replace('.', '_').replace(' ', '_')
        file_path = os.path.join(view_sql_path, f"{safe_view_name}.sql")

        # Write to .sql file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(formatted_sql)

    print(f"SQL files saved at {view_sql_path}")
    return view_sql_path

# Step 2: Scan each .sql file for table/column names
def scan_columns(table_column_file, view_sql_path, output_file):
    df_columns = pd.read_excel(table_column_file)
    results = []

    for _, col_row in df_columns.iterrows():
        table = col_row['TABLE_NAME'].upper()
        column = col_row['COLUMN_NAME'].upper()

        for view_file in os.listdir(view_sql_path):
            if view_file.endswith('.sql'):
                file_path = os.path.join(view_sql_path, view_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Check occurrences for column
                col_matching_lines = [str(i + 1) for i, line in enumerate(lines) if column in line.upper()]
                col_count = len(col_matching_lines)

                # Check occurrences for table
                tbl_matching_lines = [str(i + 1) for i, line in enumerate(lines) if table in line.upper()]
                tbl_count = len(tbl_matching_lines)

                # Save result if found
                if col_count > 0 or tbl_count > 0:
                    results.append({
                        'TABLE_NAME': table,
                        'COLUMN_NAME': column,
                        'VIEW_NAME where column found': view_file,
                        'number of times column found': col_count,
                        'column found in line numbers': ', '.join(col_matching_lines) if col_count > 0 else '',
                        'number of times table found': tbl_count,
                        'table found in line numbers': ', '.join(tbl_matching_lines) if tbl_count > 0 else ''
                    })

    df_results = pd.DataFrame(results)
    df_results.to_excel(output_file, index=False)
    print(f"Search complete! Results saved to {output_file}")

# Main execution
def main_run():
    root = Tk()
    root.withdraw()

    # Step 1: Select SQL CSV file
    sql_csv_file = gui_selection('file', 'Select SQL CSV file (with OBJECT_NAME & DDL)')
    path = os.path.dirname(sql_csv_file)

    # Split SQL files
    view_sql_path = split_sql_files(sql_csv_file, path)

    # Step 2: Select table-column Excel
    table_column_file = gui_selection('file', 'Select Table-Column Excel File')

    # Output file
    output_file = os.path.join(path, "column_view_usage_with_lines.xlsx")

    # Scan columns
    scan_columns(table_column_file, view_sql_path, output_file)

if __name__ == "__main__":
    print("**** Process started at:", datetime.datetime.now())
    main_run()
    print("**** Process completed at:", datetime.datetime.now())
  
