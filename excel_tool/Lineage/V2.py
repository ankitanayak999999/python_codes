import pandas as pd
import os
from collections import defaultdict

def generate_lineage_output(input_path):
    df = pd.read_excel(input_path)

    # Create source/target full names
    df["Source Full Name"] = df["Source DB Name"] + "." + df["Source Schema Name"] + "." + df["SourceTableName"]
    df["Target Full Name"] = df["Target DB Name"] + "." + df["Target Schema Name"] + "." + df["Target Table Name"]

    # Build lineage graph
    forward_map = defaultdict(list)
    for _, row in df.iterrows():
        forward_map[row["Source Full Name"]].append(row["Target Full Name"])

    all_sources = set(df["Source Full Name"])
    all_targets = set(df["Target Full Name"])
    root_sources = all_sources - all_targets

    paths = []

    def dfs(path):
        current = path[-1]
        if current not in forward_map:
            paths.append(path)
            return
        for nxt in forward_map[current]:
            if nxt not in path:
                dfs(path + [nxt])

    for root in root_sources:
        dfs([root])

    lineage_rows = []
    for path in paths:
        for i in range(len(path) - 1):
            lineage_rows.append({
                "Ultimate Root Source": path[0],
                "Final Target": path[-1],
                "Lineage Path": ">>".join(path[:i+2]),
                "Hop Count": len(path) - 1,
                "Node Level": f"{i+1}-{i+2}",
                "Is Leaf Node": "TRUE" if i == len(path) - 2 else "FALSE",
                "Source Full Name": path[i],
                "Target Full Name": path[i+1],
            })

    lineage_df = pd.DataFrame(lineage_rows)

    # Merge to get original metadata
    merged = pd.merge(lineage_df, df, how="left", on=["Source Full Name", "Target Full Name"])

    final_cols = [
        "Source Full Name", "Target Full Name",
        "Ultimate Root Source", "Final Target",
        "Lineage Path", "Hop Count", "Node Level", "Is Leaf Node",
        "Source DB Name", "Source Schema Name", "SourceTableName",
        "Target DB Name", "Target Schema Name", "Target Table Name"
    ]
    final_df = merged[final_cols]

    output_file = f"Result_{os.path.splitext(os.path.basename(input_path))[0]}.xlsx"
    final_df.to_excel(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")

# Example usage (remove if using GUI)
# generate_lineage_output("your_file.xlsx")
