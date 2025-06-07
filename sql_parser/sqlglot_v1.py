import sqlglot
from sqlglot.expressions import Column, Table
import pandas as pd

def extract_column_usage(sql):
    parsed = sqlglot.parse_one(sql)
    results = []

    # Step 1: Collect CTE definitions
    cte_definitions = {}
    if parsed.args.get("with"):
        for cte in parsed.args["with"].expressions:
            cte_name = cte.alias
            cte_definitions[cte_name] = cte.this

    # Step 2: Recursive helper to process expressions
    def process_expression(expr, source_level, outer_aliases=None):
        table_aliases = outer_aliases or {}

        # Resolve table aliases
        for table in expr.find_all(Table):
            table_name = table.name
            alias = table.alias_or_name
            table_aliases[alias] = table_name

        # Column usage
        for col in expr.find_all(Column):
            column_name = col.name
            table_ref = col.table
            resolved_table = table_aliases.get(table_ref, table_ref)

            # Detect context
            location = "UNKNOWN"
            parent = col.parent
            while parent:
                if parent.args.get("expressions"):
                    location = "SELECT"
                    break
                elif parent.key == "on":
                    location = "JOIN"
                    break
                elif parent.key == "where":
                    location = "WHERE"
                    break
                parent = parent.parent

            results.append({
                "table_name": resolved_table,
                "column_name": column_name,
                "column_location": location,
                "source_level": source_level
            })

        # Process nested CTEs if any
        if expr.args.get("with"):
            for inner_cte in expr.args["with"].expressions:
                cte_name = inner_cte.alias
                cte_definitions[cte_name] = inner_cte.this
                process_expression(inner_cte.this, source_level="CTE", outer_aliases=table_aliases.copy())

    # Step 3: Process CTEs
    for cte_expr in cte_definitions.values():
        process_expression(cte_expr, source_level="CTE")

    # Step 4: Process main query
    process_expression(parsed, source_level="MAIN")

    return pd.DataFrame(results)
sql = """
WITH recent_orders AS (
    SELECT order_id, customer_id, order_date
    FROM orders
    WHERE order_date > '2023-01-01'
),
customer_info AS (
    SELECT customer_id, customer_name
    FROM customers
)
SELECT r.order_id, c.customer_name
FROM recent_orders r
JOIN customer_info c ON r.customer_id = c.customer_id
"""

df = extract_column_usage(sql)
print(df)
