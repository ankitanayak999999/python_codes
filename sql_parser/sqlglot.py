import sqlglot
from sqlglot.expressions import Table, Column, CTE
import pandas as pd
import re

def remove_sql_comments(sql_text):
    """Remove single-line and multi-line comments from SQL."""
    sql_text = re.sub(r'--.*?$', '', sql_text, flags=re.MULTILINE)
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
    return sql_text

def extract_ctes_and_sources(parsed):
    """Map CTE names to their source base tables."""
    cte_table_map = {}
    for cte in parsed.find_all(CTE):
        cte_name = cte.alias_or_name.lower()
        inner_tables = {
            t.name.lower() for t in cte.this.find_all(Table)
        }
        cte_table_map[cte_name] = inner_tables
    return cte_table_map

def extract_used_columns(parsed, cte_table_map):
    """Extract used columns from base tables (trace through CTEs)."""
    used_columns = set()
    base_tables = {
        (t.alias_or_name or t.name).lower(): t.name.lower()
        for t in parsed.find_all(Table)
    }

    for col in parsed.find_all(Column):
        table_alias = (col.table or "").lower()
        column_name = col.name

        if table_alias:
            # If alias is a CTE → resolve to its source base tables
            if table_alias in cte_table_map:
                for base_table in cte_table_map[table_alias]:
                    used_columns.add((base_table, column_name))
            else:
                resolved_table = base_tables.get(table_alias, table_alias)
                used_columns.add((resolved_table, column_name))
        elif len(base_tables) == 1:
            # Single-table query, unqualified column → assign directly
            only_table = list(base_tables.values())[0]
            used_columns.add((only_table, column_name))

    return used_columns

def process_sql_file(file_path):
    """Main function to process a SQL file and return table-column usage."""
    with open(file_path, 'r') as f:
        raw_sql = f.read()

    clean_sql = remove_sql_comments(raw_sql)
    queries = sqlglot.parse(clean_sql)

    all_used = set()

    for parsed in queries:
        cte_table_map = extract_ctes_and_sources(parsed)
        used = extract_used_columns(parsed, cte_table_map)
        all_used.update(used)

    df = pd.DataFrame(sorted(all_used), columns=["TABLE_NAME", "COLUMN_NAME"])
    return df
