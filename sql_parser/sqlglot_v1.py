import sqlglot
from sqlglot.expressions import Column, Table, Expression, Star, Subquery
import pandas as pd

def extract_column_usage_corrected(sql):
    parsed = sqlglot.parse_one(sql)
    results = []
    seen = set()
    processed_nodes = set()
    cte_definitions = {}
    cte_to_base_table = {}

    # Step 1: Capture CTEs and resolve their source tables
    if parsed.args.get("with"):
        for cte in parsed.args["with"].expressions:
            cte_name = cte.alias
            cte_expr = cte.this
            cte_definitions[cte_name] = cte_expr

            tables = list(cte_expr.find_all(Table))
            if tables:
                base_table = tables[0].name  # Assume single base table
                cte_to_base_table[cte_name] = base_table

    # Step 2: Determine usage location
    def get_column_location(expr):
        parent = expr.parent
        while parent:
            if isinstance(parent, Expression):
                class_name = parent.__class__.__name__.lower()
                if parent.key == "expressions" and class_name == "select":
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

    # Step 3: Process SQL tree
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
            resolved_table = cte_to_base_table.get(resolved_table, resolved_table)
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
            resolved_table = cte_to_base_table.get(resolved_table, resolved_table)
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

    # Step 4: Process CTEs and Main
    for cte_expr in cte_definitions.values():
        process_expression(cte_expr, "CTE")

    process_expression(parsed, "MAIN")

    # Step 5: Group and merge
    df = pd.DataFrame(results)
    df_grouped = (
        df.groupby(["table_name", "column_name", "source_level"])["column_location"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .reset_index()
    )
    return df_grouped

# ========= ✅ Test SQL =========
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

    df = extract_column_usage_corrected(sql)
    print(df)
    df.to_excel("corrected_column_lineage.xlsx", index=False)
