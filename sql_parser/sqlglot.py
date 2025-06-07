import sqlglot
from sqlglot.expressions import Table, Column, CTE
import pandas as pd

def extract_cte_names(parsed):
    return {cte.alias_or_name.lower() for cte in parsed.find_all(CTE)}

def extract_used_columns(parsed, cte_names):
    used_columns = set()
    base_tables = set()

    for table in parsed.find_all(Table):
        table_name = table.name.lower()
        if table_name not in cte_names:
            base_tables.add(table.alias_or_name or table.name)

    for col in parsed.find_all(Column):
        table = col.table or ''
        column = col.name
        if table:
            used_columns.add((table.lower(), column))
        elif len(base_tables) == 1:
            only_table = list(base_tables)[0]
            used_columns.add((only_table.lower(), column))

    return used_columns

def process_sql_file(file_path):
    all_results = set()

    with open(file_path, 'r') as f:
        sql_text = f.read()

    queries = sqlglot.parse(sql_text)

    for parsed in queries:
        cte_names = extract_cte_names(parsed)
        used_columns = extract_used_columns(parsed, cte_names)
        all_results.update(used_columns)

    # Convert to DataFrame
    df = pd.DataFrame(sorted(all_results), columns=["Table Name", "Column Name"])
    return df
