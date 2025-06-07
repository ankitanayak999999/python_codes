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
WITH temp AS (
  SELECT * FROM ENTP_PREPRD_DL_DB.ENTP_PL_SCH.PL_DEPOSIT_ACCOUNT
  WHERE AS_OF_DATE = (SELECT MAX(AS_OF_DATE) 
                      FROM ENTP_PREPRD_DL_DB.ENTP_PL_SCH.PL_DEPOSIT_ACCOUNT)
)
SELECT
  A.EXTERNAL_IDENTIFIER,
  A.ID,
  B.TD_ACCT_NBR
FROM temp A
JOIN ENTP_PRD_LNDG_DB.ENTP_FIS_SCH.TB_DP_OZP_TD_ARD B
  ON A.EXTERNAL_IDENTIFIER = B.TD_ID
WHERE A.ID IS NOT NULL;
"""

df_result = extract_table_column_pairs(sql)
df_result.to_excel("final_output.xlsx", index=False)
print(df_result)
