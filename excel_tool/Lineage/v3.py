import pandas as pd
import networkx as nx
import os
from tkinter import filedialog, Tk
from datetime import datetime

def browse_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Input Excel File", 
        filetypes=[("Excel files", "*.xlsx")]
    )
    return file_path

def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        src = (row['Source DB Name'], row['Source Schema Name'], row['SourceTableName'])
        tgt = (row['Target DB Name'], row['Target Schema Name'], row['Target Table Name'])
        G.add_edge(src, tgt)
    return G

def get_leaf_nodes(G):
    return [n for n in G.nodes if G.out_degree(n) == 0]

def get_root_nodes(G):
    return [n for n in G.nodes if G.in_degree(n) == 0]

def generate_lineage_paths(G):
    roots = get_root_nodes(G)
    leaves = get_leaf_nodes(G)
    all_paths = []
    for root in roots:
        for leaf in leaves:
            if nx.has_path(G, root, leaf):
                for path in nx.all_simple_paths(G, source=root, target=leaf):
                    all_paths.append(path)
    return all_paths

def expand_paths_to_rows(paths, df):
    rows = []
    for path in paths:
        ultimate_root = '.'.join(path[0])
        final_target = '.'.join(path[-1])
        lineage_path = ' >> '.join(['.'.join(p) for p in path])
        hop_count = len(path) - 1

        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]
            node_level = f"{i+1}-{i+2}"
            is_leaf_node = (i + 1 == len(path) - 1)

            match_row = df[
                (df['Source DB Name'] == src[0]) &
                (df['Source Schema Name'] == src[1]) &
                (df['SourceTableName'] == src[2]) &
                (df['Target DB Name'] == tgt[0]) &
                (df['Target Schema Name'] == tgt[1]) &
                (df['Target Table Name'] == tgt[2])
            ]

            if not match_row.empty:
                base = match_row.iloc[0].to_dict()
                base.update({
                    'Ultimate Root Source': ultimate_root,
                    'Final Target': final_target,
                    'Lineage Path': lineage_path,
                    'Hop Count': hop_count,
                    'Node Level': node_level,
                    'Is Leaf Node': is_leaf_node
                })
                rows.append(base)
    return pd.DataFrame(rows)

def save_output(df, suffix="lineage_output"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), "lineage_results")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{suffix}_{timestamp}.xlsx")
    df.to_excel(file_path, index=False)
    return file_path

def run_lineage_workflow():
    input_file = browse_file()
    if not input_file:
        print("❌ No file selected.")
        return

    df = pd.read_excel(input_file)

    required_cols = [
        'Source DB Name', 'Source Schema Name', 'SourceTableName',
        'Target DB Name', 'Target Schema Name', 'Target Table Name'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return

    G = build_graph(df)
    paths = generate_lineage_paths(G)
    output_df = expand_paths_to_rows(paths, df)
    output_file = save_output(output_df)
    print(f"\n✅ Lineage file saved to:\n{output_file}")

# Run it
run_lineage_workflow()
