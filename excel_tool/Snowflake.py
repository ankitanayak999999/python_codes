from snowflake.snowpark import Session

connection_parameters = {
    "account": "<your_account>",
    "user": "<your_username>",
    "role": "<your_role>",
    "warehouse": "<your_warehouse>",
    "database": "<your_database>",
    "schema": "<your_schema>",
    "authenticator": "externalbrowser"
}

session = Session.builder.configs(connection_parameters).import snowflake.connector

conn = snowflake.connector.connect(
    user='USER',
    account='ACCOUNT',
    authenticator='externalbrowser',
    warehouse='WH',
    database='DB',
    schema='SCHEMA'
)
cur = conn.cursor()
cur.execute("SELECT * FROM MY_TABLE LIMIT 10")
for row in cur:
    print(row)
