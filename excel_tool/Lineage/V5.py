import pandas as pd
import networkx as nx
import os

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

def extract_lineage_paths(graph, source):
    paths = []
    for node in nx.descendants(graph, source):
        if graph.out_degree(node) == 0:
            for path in nx.all_simple_paths(graph, source, node):
                paths.append(path)
    return paths

def generate_lineage_output(input_path):
    df = pd.read_excel(input_path)

    # Build full names
    df['Source Full Name'] = df['Source DB Name'] + '.' + df['Source Schema Name'] + '.' + df['SourceTableName']
    df['Target Full Name'] = df['Target DB Name'] + '.' + df['Target Schema Name'] + '.' + df['Target Table Name']
    df['Final Target'] = df['Target Full Name']

    # Build graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Source Full Name'], row['Target Full Name'])

    # Lineage path and hops
    lineage_records = []
    for source in df['Source Full Name'].unique():
        paths = extract_lineage_paths(G, source)
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
                    'Target Full Name': path[1] if len(path) > 1 else path[0],
                    'Final Target': path[-1],
                    'Lineage Path': ' → '.join([p.split('.')[-1] for p in path]),
                    'Hop Count': len(path) - 1
                })

    lineage_df = pd.DataFrame(lineage_records).drop_duplicates()

    # Merge and enrich
    merged_df = pd.merge(df, lineage_df, on=['Source Full Name', 'Target Full Name'], how='left')
    merged_df['Is Leaf Node'] = merged_df['Target Full Name'].apply(lambda x: G.out_degree(x) == 0)
    merged_df['Final Target'] = merged_df['Final Target'].astype(str)
    merged_df['Ultimate Root Source'] = merged_df['Final Target'].apply(lambda x: trace_upstream_true_roots(x, G))
    exploded_df = merged_df.explode('Ultimate Root Source')

    # Create final output
    exploded_df['Full Source Table Name'] = exploded_df['Source Full Name']
    exploded_df['Full Target Table Name'] = exploded_df['Final Target']

    final_columns = [
        'Full Source Table Name', 'Full Target Table Name',
        'Ultimate Root Source', 'Source Full Name', 'Final Target',
        'Lineage Path', 'Hop Count', 'Is Leaf Node',
        'Source DB Name', 'Source Schema Name', 'SourceTableName',
        'Target DB Name', 'Target Schema Name', 'Target Table Name'
    ]

    final_df = exploded_df[final_columns]

    # Export to Excel
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"Result_{input_filename}.xlsx"
    final_df.to_excel(output_path, index=False)

    print(f"✅ Lineage result saved to: {output_path}")

if __name__ == "__main__":
    # Example call: just replace this with your actual filename
    generate_lineage_output("Lineage_with_Leaf_Node_Flag.xlsx")
