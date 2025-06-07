import sqlglot
from sqlglot.expressions import Column, Table, Expression, Star, Subquery
import pandas as pd

def extract_column_usage_with_fixed_cte(sql):
    parsed = sqlglot.parse_one(sql)
    results = []
    seen = set()
    processed_nodes = set()
    cte_sources = {}

    # Step 1: Capture CTEs and their source tables
    cte_definitions = {}
    if parsed.args.get("with"):
        for cte in parsed.args["with"].expressions:
            cte_name = cte.alias
            cte_definitions[cte_name] = cte.this
            sources = {}
            for table in cte.this.find_all(Table):
                alias = table.alias_or_name
                sources[alias] = table.name
            cte_sources[cte_name] = sources

    # Step 2: Determine column usage location
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

    # Step 3: Parse expressions recursively
    def process_expression(expr, source_level, outer_aliases=None):
        if id(expr) in processed_nodes:
            return
        processed_nodes.add(id(expr))

        table_aliases = outer_aliases or {}

        for table in expr.find_all(Table):
            table_aliases[table.alias_or_name] = table.name

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

        for subquery in expr.find_all(Subquery):
            process_expression(subquery, source_level, table_aliases.copy())

    # Step 4: Process all CTEs and then the main query
    for cte_name, cte_expr in cte_definitions.items():
        process_expression(cte_expr, "CTE")

    process_expression(parsed, "MAIN")

    # Step 5: Group and combine column locations
    df = pd.DataFrame(results)
    df_grouped = (
        df.groupby(["table_name", "column_name", "source_level"])["column_location"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .reset_index()
    )
    return df_grouped

# ========== ✅ Example SQL ==========

if __name__ == "__main__":
    sql = """
    WITH recent_orders AS (
        SELECT * FROM orders WHERE order_date > '2023-01-01'
    )
    SELECT r.order_id
    FROM recent_orders r
    JOIN customers c ON r.customer_id = c.customer_id
    WHERE c.region = 'East'
    """

    df = extract_column_usage_with_fixed_cte(sql)
    print(df)
    df.to_excel("column_lineage_cte_where_fixed.xlsx", index=False)
