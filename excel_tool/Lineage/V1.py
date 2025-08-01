import pandas as pd
import networkx as nx
import os
import sys
import tkinter as tk
from tkinter import filedialog

def trace_upstream_true_roots_list(node, graph):
    roots = set()
    stack = [node]
    visited = set()
    while stack:
        current = stack.pop()
        visited.add(current)
        preds = list(graph.predecessors(current))
        if not preds:
            roots.add(current)
        else:
            for pred in preds:
                if pred not in visited:
                    stack.append(pred)
    return sorted(roots) if roots else []

def extract_lineage_summary(graph, df):
    lineage_data = []

    for _, row in df.iterrows():
        src = row['Source Full Name']
        tgt = row['Target Full Name']

        # Trace full path to leaf
        paths = list(nx.all_simple_paths(graph, src, tgt)) if tgt in nx.descendants(graph, src) else []

        if not paths:
            # Just source → target
            lineage_path = f"{src.split('.')[-1]} → {tgt.split('.')[-1]}"
            hop_count = 1
            node_level = "Node 1-2"
            final_target = tgt
        else:
            # Take first path
            path = paths[0]
            lineage_path = ' → '.join([p.split('.')[-1] for p in path])
            hop_count = len(path) - 1
            node_level = f"Node 1-{hop_count + 1}"
            final_target = path[-1]

        # Compute leaf and root(s)
        is_leaf = graph.out_degree(tgt) == 0
        root_sources = trace_upstream_true_roots_list(final_target, graph)
        ultimate_root = ', '.join(root_sources)

        lineage_data.append({
            'Source Full Name': src,
            'Target Full Name': tgt,
            'Final Target': final_target,
            'Lineage Path': lineage_path,
            'Hop Count': hop_count,
            'Node Level': node_level,
            'Is Leaf Node': is_leaf,
            'Ultimate Root Source': ultimate_root
        })

    return pd.DataFrame(lineage_data)

def generate_lineage_output(input_path):
    df = pd.read_excel(input_path)

    # Build full names
    df['Source Full Name'] = df['Source DB Name'] + '.' + df['Source Schema Name'] + '.' + df['SourceTableName']
    df['Target Full Name'] = df['Target DB Name'] + '.' + df['Target Schema Name'] + '.' + df['Target Table Name']

    # Build graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Source Full Name'], row['Target Full Name'])

    # Compute lineage summary (1 row per original row)
    lineage_summary = extract_lineage_summary(G, df)

    # Merge back to original
    merged_df = pd.concat([df, lineage_summary.drop(columns=['Source Full Name', 'Target Full Name'])], axis=1)
    merged_df['Full Source Table Name'] = df['Source Full Name']
    merged_df['Full Target Table Name'] = df['Target Full Name']

    # Final column ordering
    final_columns = [
        'Full Source Table Name', 'Full Target Table Name',
        'Ultimate Root Source', 'Source Full Name', 'Final Target',
        'Lineage Path', 'Hop Count', 'Node Level', 'Is Leaf Node',
        'Source DB Name', 'Source Schema Name', 'SourceTableName',
        'Target DB Name', 'Target Schema Name', 'Target Table Name'
    ]
    final_df = merged_df[final_columns]

    # Save
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, f"Result_Grouped_{input_filename}.xlsx")
    final_df.to_excel(output_path, index=False)
    print(f"\n✅ Grouped-root lineage result saved to: {output_path}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    input_file = filedialog.askopenfilename(
        title="Select your Excel Lineage File",
        filetypes=[("Excel Files", "*.xlsx *.xls")]
    )

    if not input_file:
        print("❌ No file selected. Exiting.")
        sys.exit(1)

    generate_lineage_output(input_file)
