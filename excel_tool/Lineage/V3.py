import pandas as pd
import networkx as nx
import os
from tkinter import Tk, filedialog

def find_true_roots(node, G_rev, valid_roots):
    roots = set()
    stack = [node]
    visited = set()
    while stack:
        current = stack.pop()
        visited.add(current)
        preds = list(G_rev.predecessors(current))
        if not preds and current in valid_roots:
            roots.add(current)
        else:
            for pred in preds:
                if pred not in visited:
                    stack.append(pred)
    return sorted(roots)

def find_lineage_paths(graph, source):
    paths = []
    for node in nx.descendants(graph, source):
        if graph.out_degree(node) == 0:
            for path in nx.all_simple_paths(graph, source=source, target=node):
                paths.append(path)
    return paths

def generate_lineage(input_file: str) -> pd.DataFrame:
    df = pd.read_excel(input_file)

    # Create full source and target names
    df['Source Full Name'] = df['Source DB Name'] + '.' + df['Source Schema Name'] + '.' + df['SourceTableName']
    df['Target Full Name'] = df['Target DB Name'] + '.' + df['Target Schema Name'] + '.' + df['Target Table Name']

    # Build graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Source Full Name'], row['Target Full Name'])
    G_rev = G.reverse()

    # Determine true ultimate roots
    true_roots = sorted(set(df['Source Full Name']) - set(df['Target Full Name']))

    # Generate lineage paths
    lineage_records = []
    for source in df['Source Full Name'].unique():
        paths = find_lineage_paths(G, source)
        if not paths:
            targets = df[df['Source Full Name'] == source]['Target Full Name'].unique()
            for target in targets:
                lineage_records.append({
                    'Source Full Name': source,
                    'Target Full Name': target,
                    'Final Target': target,
                    'Lineage Path': f"{source.split('.')[-1]} → {target.split('.')[-1]}",
                    'Hop Count': 1
                })
        else:
            for path in paths:
                lineage_records.append({
                    'Source Full Name': path[0],
                    'Target Full Name': path[1],
                    'Final Target': path[-1],
                    'Lineage Path': ' → '.join([p.split('.')[-1] for p in path]),
                    'Hop Count': len(path) - 1
                })

    lineage_df = pd.DataFrame(lineage_records).drop_duplicates()

    # Merge and enrich
    df_merged = pd.merge(df, lineage_df, on=['Source Full Name', 'Target Full Name'], how='left')
    df_merged['Is Leaf Node'] = df_merged['Target Full Name'].apply(lambda x: G.out_degree(x) == 0)

    # Find all real root sources
    df_merged['All Ultimate Roots'] = df_merged['Source Full Name'].apply(
        lambda x: find_true_roots(x, G_rev, true_roots)
    )

    # Explode into one row per root
    df_final = df_merged.explode('All Ultimate Roots').rename(columns={'All Ultimate Roots': 'Ultimate Root Source'})

    # Reorder columns
    lineage_cols = [
        'Ultimate Root Source', 'Source Full Name', 'Target Full Name',
        'Final Target', 'Lineage Path', 'Hop Count', 'Is Leaf Node'
    ]
    other_cols = [col for col in df.columns if col not in lineage_cols]
    final_df = df_final[lineage_cols + other_cols]

    # Save output
    base_name = os.path.basename(input_file)
    output_path = os.path.join(os.path.dirname(input_file), f"Result_{os.path.splitext(base_name)[0]}.xlsx")
    final_df.to_excel(output_path, index=False)
    print(f"\n✅ Lineage file saved as: {output_path}")
    return final_df

if __name__ == "__main__":
    Tk().withdraw()
    input_file = filedialog.askopenfilename(title="Select your lineage Excel file", filetypes=[("Excel files", "*.xlsx")])
    if not input_file:
        print("❌ No file selected.")
    else:
        generate_lineage(input_file)ex
