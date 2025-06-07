####v1
import sqlglot
from sqlglot.expressions import Table, Column, CTE
import pandas as pd
import re

def remove_sql_comments(sql_text):
    sql_text = re.sub(r'--.*?$', '', sql_text, flags=re.MULTILINE)
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
    return sql_text

def extract_cte_sources(parsed):
    cte_sources = {}
    for cte in parsed.find_all(CTE):
        cte_name = cte.alias_or_name.lower()
        base_tables = {t.name for t in cte.this.find_all(Table)}
        cte_sources[cte_name] = base_tables
    return cte_sources

def resolve_columns_to_base_tables(parsed, cte_sources):
    used_columns = set()
    alias_to_table = {}

    for tbl in parsed.find_all(Table):
        alias = (tbl.alias_or_name or tbl.name).lower()
        alias_to_table[alias] = tbl.name

    for col in parsed.find_all(Column):
        if col.name == "*":
            continue
        table_alias = (col.table or "").lower()
        col_name = col.name

        if table_alias in alias_to_table:
            actual = alias_to_table[table_alias]
            if actual.lower() in cte_sources:
                for src in cte_sources[actual.lower()]:
                    used_columns.add((src, col_name))
            else:
                used_columns.add((actual, col_name))
        elif len(alias_to_table) == 1:
            only_table = list(alias_to_table.values())[0]
            if only_table.lower() not in cte_sources:
                used_columns.add((only_table, col_name))

    return used_columns

def process_sql_file(file_path):
    with open(file_path, 'r') as f:
        sql = f.read()

    sql_clean = remove_sql_comments(sql)
    queries = sqlglot.parse(sql_clean)
    final_used = set()

    for parsed in queries:
        cte_sources = extract_cte_sources(parsed)
        final_used.update(resolve_columns_to_base_tables(parsed, cte_sources))

    return pd.DataFrame(sorted(final_used), columns=["TABLE_NAME", "COLUMN_NAME"])
