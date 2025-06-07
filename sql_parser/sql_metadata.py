from sql_metadata import Parser
import pandas as pd
import re

def remove_sql_comments(sql_text):
    """Remove SQL comments (single-line and multi-line)."""
    sql_text = re.sub(r'--.*?$', '', sql_text, flags=re.MULTILINE)
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
    return sql_text

def extract_table_column_pairs(sql_text):
    sql_text = remove_sql_comments(sql_text)
    parser = Parser(sql_text)

    tables = parser.tables
    columns = parser.columns_dict.get('select', []) + \
              parser.columns_dict.get('where', []) + \
              parser.columns_dict.get('join', []) + \
              parser.columns_dict.get('group_by', []) + \
              parser.columns_dict.get('having', []) + \
              parser.columns_dict.get('order_by', [])

    table_column_pairs = set()
    for table in tables:
        for column in columns:
            table_column_pairs.add((table, column))

    df = pd.DataFrame(sorted(table_column_pairs), columns=["TABLE_NAME", "COLUMN_NAME"])
    return df

# EXAMPLE USAGE
sql = """

"""

df_result = extract_table_column_pairs(sql)
df_result.to_excel("final_output.xlsx", index=False)
print(df_result)
