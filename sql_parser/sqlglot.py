
import sqlglot
from sqlglot.expressions import Table, Column, CTE
import pandas as pd

def extract_cte_names(parsed):
    return {cte.alias_or_name for cte in parsed.find_all(CTE)}

def extract_base_tables(parsed, cte_names):
    return {
        table.alias_or_name or table.name
        for table in parsed.find_all(Table)
        if table.alias_or_name not in cte_names and table.name not in cte_names
    }

def extract_columns(parsed, base_tables, select_star):
    table_columns = []

    if select_star:
        for table in base_tables:
            table_columns.append((table, "All"))
    else:
        for col in parsed.find_all(Column):
            if col.table in base_tables:
                table_columns.append((col.table, col.name))
            elif not col.table and len(base_tables) == 1:
                # No table alias: assume it's from the only base table
                table = list(base_tables)[0]
                table_columns.append((table, col.name))
    return table_columns

def analyze_sql(sql):
    parsed = sqlglot.parse_one(sql)
    cte_names = extract_cte_names(parsed)
    base_tables = extract_base_tables(parsed, cte_names)

    select_star = any(token.text == '*' for token in parsed.find_all())
    return extract_columns(parsed, base_tables, select_star)

def process_sql_file(file_path):
    with open(file_path, "r") as f:
        sql_text = f.read()

    queries = sqlglot.parse(sql_text)
    all_results = []

    for parsed in queries:
        sql = parsed.sql()
        results = analyze_sql(sql)
        all_results.extend(results)

    return all_results

# 📂 Input SQL file path
input_file = "input.sql"  # Replace with your file name

# 🔍 Process and extract
table_column_data = process_sql_file(input_file)

# 📊 Export to Excel
df = pd.DataFrame(table_column_data, columns=["Table Name", "Column Name"])
df.to_excel("sqlglot_table_column_output.xlsx", index=False)

print("✅ Exported to 'sqlglot_table_column_output.xlsx'")
