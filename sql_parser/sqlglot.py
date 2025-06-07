import sqlglot
from sqlglot.expressions import Table, Column, CTE
import pandas as pd
import re

def remove_sql_comments(sql_text):
    """Remove -- and /* */ comments from SQL."""
    sql_text = re.sub(r'--.*?$', '', sql_text, flags=re.MULTILINE)
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
    return sql_text

def extract_cte_definitions(parsed):
    """Map CTE names to the base tables used inside them."""
    cte_sources = {}
    for cte in parsed.find_all(CTE):
        cte_name = cte.alias_or_name.lower()
        base_tables = {t.name.lower() for t in cte.this.find_all(Table)}
        cte_sources[cte_name] = base_tables
    return cte_sources

def extract_columns_from_base_tables(parsed, cte_sources):
    """Trace used columns back to original base tables."""
    used = set()
    alias_to_table = {}
    cte_names = set(cte_sources.keys())

    for table in parsed.find_all(Table):
        alias = (table.alias_or_name or table.name).lower()
        actual_name = table.name.lower()
        alias_to_table[alias] = actual_name

    for col in parsed.find_all(Column):
        col_name = col.name
        if col_name == "*":
            continue  # ignore unqualified SELECT *
        table_alias = (col.table or "").lower()

        if table_alias in alias_to_table:
            actual = alias_to_table[table_alias]
            if actual in cte_names:
                for real_base in cte_sources[actual]:
                    used.add((real_base, col_name))
            else:
                used.add((actual, col_name))
        elif len(alias_to_table) == 1:
            only_table = list(alias_to_table.values())[0]
            if only_table not in cte_names:
                used.add((only_table, col_name))

    return used

def process_sql_file(file_path):
    """Main function: clean, parse, extract used columns from real tables."""
    with open(file_path, 'r') as f:
        sql = f.read()

    clean_sql = remove_sql_comments(sql)
    queries = sqlglot.parse(clean_sql)
    final_used = set()

    for parsed in queries:
        cte_sources = extract_cte_definitions(parsed)
        used = extract_columns_from_base_tables(parsed, cte_sources)
        final_used.update(used)

    df = pd.DataFrame(sorted(final_used), columns=["TABLE_NAME", "COLUMN_NAME"])
    return df
