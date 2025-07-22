import pandas as pd

# Input files
file1 = "file1.xlsx"  # File with list of table names
file2 = "file2.xlsx"  # File with multiple columns
output_file = "file2_with_table_match.xlsx"

# Read Excel files
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Ensure file1 has a column name
df1.columns = ['TableName']

# Convert table names to uppercase list for substring matching
table_list = df1['TableName'].dropna().astype(str).str.upper().tolist()

# ---- Columns to check ----
# Example: columns_to_check = ['Col1', 'Col2', 'Col3', 'Col4']
# If not a list, it will automatically check all columns
columns_to_check = "ALL"  # or ['Col1','Col2']

# Ensure columns_to_check is a list, otherwise select all columns
if not isinstance(columns_to_check, list):
    columns_to_check = df2.columns.tolist()

# Function to find all tables from File 1 present in the selected columns (substring search)
def find_table_in_row(row):
    found_tables = []
    row_text = " ".join([str(row[col]).upper() for col in columns_to_check if pd.notna(row[col])])
    for table in table_list:
        if table in row_text:
            found_tables.append(table)
    return ", ".join(found_tables) if found_tables else "NOT FOUND"

# Apply function to each row (only using the selected columns for comparison)
df2['MatchedTables'] = df2.apply(find_table_in_row, axis=import re

def find_table_in_row(row):
    found_tables = []
    # Combine selected columns into a single string (uppercase)
    row_text = " ".join([str(row[col]).upper() for col in columns_to_check if pd.notna(row[col])])
    
    for table in table_list:
        # Match table name, but ensure no underscore before or after
        pattern = r'(?<!_)' + re.escape(table) + r'(?!_)'
        if re.search(pattern, row_text):
            found_tables.append(table)
    
    return ", ".join(found_tables) if found_tables else None



# Save updated File 2
df2.to_excel(output_file, index=False)

print(f"Updated file2 saved as {output_file}")
