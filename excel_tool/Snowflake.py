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

session = Session.builder.configs(connection_parameters).create()
