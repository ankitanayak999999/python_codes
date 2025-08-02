import pandas as pd
import networkx as nx

# Construct DataFrame from manually captured screenshot data
data = {
    "Source DB Name": [
        "ENTP_DEV_STG_DB", "ENTP_DEV_STG_DB", "ENTP_DEV_STG_DB", "ENTP_DEV_STG_DB",
        "ENTP_DEV_STG_DB", "ENTP_DEV_STG_DB", "ENTP_DEV_STG_DB", "ENTP_DEV_STG_DB",
        "ENTP_DEV_STG_DB", "ENTP_DEV_STG_DB"
    ],
    "Source Schema Name": ["ENTP_WRK_SCH"] * 10,
    "SourceTableName": [
        "RS_ADDR", "RS_ADDR", "RS_ADDR_V_1", "RS_COMBINE_V", "RS_COMBINE_V_1",
        "RS_ADDR", "RS_ACCT", "RS_ACCT", "RS_ACCT", "RS_ACCT"
    ],
    "Target DB Name": ["ENTP_DEV_STG"] * 10,
    "Target Schema Name": ["ENTP_WRK_SCH"] * 10,
    "Target Table Name": [
        "RS_ADDR_V_1", "RS_COMBINE_V", "RS_ADDR_V_3", "RS_COMBINE_V_1", "RS_COMBINE_V_2",
        "RS_COMBINE_V_3", "RS_COMBINE_V_2", "RS_COMBINE_V", "RS_ACCT_V_1", "RS_COMBINE_V_2"
    ]
}

df = pd.DataFrame(data)

# Build the directed lineage graph
def build_lineage_graph(df):
    g = nx.DiGraph()
    for _, row in df.iterrows():
        source = f"{row['Source DB Name']}.{row['Source Schema Name']}.{row['SourceTableName']}"
        target = f"{row['Target DB Name']}.{row['Target Schema Name']}.{row['Target Table Name']}"
        g.add_edge(source, target, metadata=row.to_dict())
    return g

# Extract paths from root to leaf and explode lineage rows
def extract_lineage_paths(graph):
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]
    leaves = [n for n in graph.nodes if graph.out_degree(n) == 0]
    all_paths = []

    for root in roots:
        for leaf in leaves:
            for path in nx.all_simple_paths(graph, source=root, target=leaf):
                for i in range(len(path) - 1):
                    src = path[i]
                    tgt = path[i+1]
                    edge_data = graph.get_edge_data(src, tgt)['metadata']
                    all_paths.append({
                        "Ultimate Root Source": root,
                        "Final Target": path[-1],
                        "Lineage Path": ">>".join(path),
                        "Hop Count": len(path) - 1,
                        "Node Level": f"{i+1}-{i+2}",
                        "Is Leaf Node": tgt == path[-1],
                        "Source DB Name": edge_data["Source DB Name"],
                        "Source Schema Name": edge_data["Source Schema Name"],
                        "SourceTableName": edge_data["SourceTableName"],
                        "Target DB Name": edge_data["Target DB Name"],
                        "Target Schema Name": edge_data["Target Schema Name"],
                        "Target Table Name": edge_data["Target Table Name"],
                    })
    return pd.DataFrame(all_paths)

# Run the full lineage extraction
graph = build_lineage_graph(df)
lineage_df = extract_lineage_paths(graph)

# Display result in dataframe
import ace_tools as tools; tools.display_dataframe_to_user(name="Final Lineage Output", dataframe=lineage_df)
