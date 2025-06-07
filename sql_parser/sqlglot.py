#v2 

import sqlglot
from sqlglot.expressions import Table, Column, CTE, Expression
import pandas as pd
import re

def remove_sql_comments(sql_text):
    """Remove SQL single-line and multi-line comments."""
    sql_text = re.sub(r'--.*?$', '', sql_text, flags=re.MULTILINE)
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
    return sql_text

def extract_cte_sources(parsed):
    """Map CTE name to underlying base tables used."""
    cte_sources = {}
    for cte in parsed.find_all(CTE):
        cte_name = cte.alias_or_name.lower()
        base_tables = {tbl.name for tbl in cte.this.find_all(Table)}
        cte_sources[cte_name] = base_tables
    return cte_sources

def find_all_columns(expr: Expression) -> set:
    """Recursively collect all Column references."""
    return set(expr.find_all(Column))

def resolve_columns_to_base_tables(parsed, cte_sources):
    """Trace columns to original physical base tables."""
    used_columns = set()
    alias_to_table = {}

    for tbl in parsed.find_all(Table):
        alias = (tbl.alias_or_name or tbl.name).lower()
        alias_to_table[alias] = tbl.name  # Keep full qualified name

    all_columns = find_all_columns(parsed)

    for col in all_columns:
        if col.name == "*":
            continue
        table_alias = (col.table or "").lower()
        col_name = col.name

        if table_alias in alias_to_table:
            actual_table = alias_to_table[table_alias]
            if actual_table.lower() in cte_sources:
                for real_base in cte_sources[actual_table.lower()]:
                    used_columns.add((real_base, col_name))
            else:
                used_columns.add((actual_table, col_name))
        elif len(alias_to_table) == 1:
            only_table = list(alias_to_table.values())[0]
            if only_table.lower() not in cte_sources:
                used_columns.add((only_table, col_name))

    return used_columns

def process_sql_file(file_path):
    """Main function to extract base table-column pairs from a SQL file."""
    with open(file_path, 'r') as f:
        sql_text = f.read()

    sql_clean = remove_sql_comments(sql_text)
    queries = sqlglot.parse(sql_clean)
    final_used = set()

    for parsed in queries:
        cte_sources = extract_cte_sources(parsed)
        resolved = resolve_columns_to_base_tables(parsed, cte_sources)
        final_used.update(resolved)

    return pd.DataFrame(sorted(final_used), columns=["TABLE_NAME", "COLUMN_NAME"])
