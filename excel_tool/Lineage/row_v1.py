import pandas as pd
import networkx as nx
import os
import sys
import tkinter as tk
from tkinter import filedialog

def trace_upstream_true_roots(node, graph):
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
    return sorted(roots) if roots else [None]

def extract_lineage_paths_with_levels(graph):
    lineage_records = []
    for source in graph.nodes():
        for target in nx.descendants(graph, source):
            if graph.out_degree(target) == 0:
                paths = list(nx.all_simple_paths(graph, source, target))
                for path in paths:
                    for i in range(len(path) - 1):
                        lineage_records.append({
                            'Source Full Name': path[i],
                            'Target Full Name': path[i + 1],
                            'Final Target': path[-1],
                            'Lineage Path': ' → '.join([p.split('.')[-1] for p in path]),
                            'Hop Count': len(path) - 1,
                            'Node Level': f'Node {i + 1}-{i + 2}'
                        })
    return pd.DataFrame(lineage_records)

def generate_lineage_output(input_path):
    df = pd.read_excel(input_path)

    df['Source Full Name'] = df['Source DB Name'] + '.' + df['Source Schema Name'] + '.' + df['SourceTableName']
    df['Target Full Name'] = df['Target DB Name'] + '.' + df['Target Schema Name'] + '.' + df['Target Table Name']
    df['Final Target'] = df['Target Full Name']

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Source Full Name'], row['Target Full Name'])

    lineage_df = extract_lineage_paths_with_levels(G).drop_duplicates()

    merged_df = pd.merge(df, lineage_df, on=['Source Full Name', 'Target Full Name'], how='left')

    merged_df['Is Leaf Node'] = merged_df['Target Full Name'].apply(lambda x: G.out_degree(x) == 0)

    merged_df['Ultimate Root Source'] = merged_df['Final Target'].apply(lambda x: trace_upstream_true_roots(x, G))
    exploded_df = merged_df.explode('Ultimate Root Source')

    final_df = exploded_df[[
        'Source Full Name', 'Target Full Name', 'Ultimate Root Source', 'Final Target',
        'Lineage Path', 'Hop Count', 'Node Level', 'Is Leaf Node',
        'Source DB Name', 'Source Schema Name', 'SourceTableName',
        'Target DB Name', 'Target Schema Name', 'Target Table Name'
    ]]

    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, f"Result_Exploded_{input_filename}.xlsx")
    final_df.to_excel(output_path, index=False)
    print(f"✅ Exploded lineage result saved to: {output_path}")

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
