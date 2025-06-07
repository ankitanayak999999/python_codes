import sqlglot
from sqlglot.expressions import Column, Table, Expression, Star, Subquery
import pandas as pd

def extract_column_usage_full(sql):
    parsed = sqlglot.parse_one(sql)
    results = []
    seen = set()
    processed_nodes = set()  # Prevent infinite recursion

    cte_definitions = {}
    cte_sources = {}

    # Step 1: Collect CTEs and their source tables
    if parsed.args.get("with"):
        for cte in parsed.args["with"].expressions:
            cte_name = cte.alias
            cte_definitions[cte_name] = cte.this
            sources = {}
            for table in cte.this.find_all(Table):
                alias = table.alias_or_name
                sources[alias] = table.name
            cte_sources[cte_name] = sources

    # Step 2: Determine where a column is used
    def get_column_location(expr):
        parent = expr.parent
        while parent:
            if isinstance(parent, Expression):
                if parent.key == "expressions" and parent.__class__.__name__ == "Select":
                    return "SELECT"
                if parent.key == "on":
                    return "JOIN"
                if parent.key == "where":
                    return "WHERE"
                if parent.key in ("group", "group_by"):
                    return "GROUP_BY"
                if parent.key in ("order", "order_by"):
                    return "ORDER_BY"
                if parent.key == "having":
                    return "HAVING"
            parent = parent.parent
        return "UNKNOWN"

    # Step 3: Main recursive logic
    def process_expression(expr, source_level, outer_aliases=None):
        if id(expr) in processed_nodes:
            return
        processed_nodes.add(id(expr))

        table_aliases = outer_aliases or {}

        # Register table aliases
        for table in expr.find_all(Table):
            alias = table.alias_or_name
            table_aliases[alias] = table.name

        # Handle SELECT * or table.*
        for star in expr.find_all(Star):
            table_ref = star.this or "UNKNOWN"
            resolved_table = table_aliases.get(table_ref, table_ref)
            location = get_column_location(star)
            key = (resolved_table, "*", location, source_level)
            if key not in seen:
                seen.add(key)
                results.append({
                    "table_name": resolved_table,
                    "column_name": "*",
                    "column_location": location,
                    "source_level": source_level
                })

        # Handle named columns
        for col in expr.find_all(Column):
            column_name = col.name
            table_ref = col.table
            resolved_table = table_aliases.get(table_ref, table_ref)
            if resolved_table in cte_sources:
                inner_map = cte_sources[resolved_table]
                resolved_table = list(inner_map.values())[0] if inner_map else resolved_table

            location = get_column_location(col)
            key = (resolved_table, column_name, location, source_level)
            if key not in seen:
                seen.add(key)
                results.append({
                    "table_name": resolved_table,
                    "column_name": column_name,
                    "column_location": location,
                    "source_level": source_level
                })

        # Recurse into inner CTEs
        if expr.args.get("with"):
            for inner_cte in expr.args["with"].expressions:
                inner_name = inner_cte.alias
                cte_definitions[inner_name] = inner_cte.this
                sources = {}
                for table in inner_cte.this.find_all(Table):
                    alias = table.alias_or_name
                    sources[alias] = table.name
                cte_sources[inner_name] = sources
                process_expression(inner_cte.this, source_level="CTE", outer_aliases=table_aliases.copy())

        # Recurse into subqueries
        for subquery in expr.find_all(Subquery):
            process_expression(subquery, source_level=source_level, outer_aliases=table_aliases.copy())

    # Step 4: Process all expressions
    for cte_expr in cte_definitions.values():
        process_expression(cte_expr, source_level="CTE")
    process_expression(parsed, source_level="MAIN")

    # Step 5: Merge column usage by location
    df = pd.DataFrame(results)
    df_grouped = (
        df.groupby(["table_name", "column_name", "source_level"])["column_location"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .reset_index()
    )
    return df_grouped

# ========== ✅ Example Usage ==========

if __name__ == "__main__":
    sql = """
    WITH orders_cte AS (
        SELECT * FROM orders WHERE order_date >= '2023-01-01'
    )
    SELECT c.customer_name, sub.total
    FROM customers c
    JOIN (
        SELECT customer_id, SUM(amount) as total
        FROM orders_cte
        GROUP BY customer_id
        HAVING SUM(amount) > 1000
    ) sub ON c.customer_id = sub.customer_id
    WHERE c.region = 'East'
    ORDER BY sub.total DESC
    """

    df = extract_column_usage_full(sql)
    print(df)

    # Save to Excel
    df.to_excel("column_lineage_final.xlsx", index=False)
