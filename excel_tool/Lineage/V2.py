import os
import pandas as pd
import networkx as nx
from tkinter import Tk, filedialog

def generate_lineage_output(input_path):
    df = pd.read_excel(input_path)

    # Add full names
    df['Source Full Name'] = df['Source DB Name'].astype(str) + '.' + df['Source Schema Name'].astype(str) + '.' + df['SourceTableName'].astype(str)
    df['Target Full Name'] = df['Target DB Name'].astype(str) + '.' + df['Target Schema Name'].astype(str) + '.' + df['Target Table Name'].astype(str)

    # Build the graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Source Full Name'], row['Target Full Name'])

    # Identify root (ultimate source) and leaf (final target) nodes
    sources = set(df['Source Full Name'])
    targets = set(df['Target Full Name'])
    root_nodes = sources - targets
    leaf_nodes = targets - sources

    # Build lineage paths
    all_paths = []
    for root in root_nodes:
        for target in targets:
            if root != target and nx.has_path(G, root, target):
                for path in nx.all_simple_paths(G, source=root, target=target):
                    for i in range(len(path) - 1):
                        src = path[i]
                        tgt = path[i + 1]
                        lineage_path = ' >> '.join(path)
                        hop_count = len(path) - 1
                        node_level = f"{i+1}-{i+2}"
                        is_leaf = tgt not in sources
                        # Get original row for metadata
                        match = df[(df['Source Full Name'] == src) & (df['Target Full Name'] == tgt)]
                        if not match.empty:
                            base = match.iloc[0].to_dict()
                            base.update({
                                'Source Full Name': src,
                                'Target Full Name': tgt,
                                'Ultimate Root Source': root,
                                'Final Target': target,
                                'Lineage Path': lineage_path,
                                'Hop Count': hop_count,
                                'Node Level': node_level,
                                'Is Leaf Node': is_leaf
                            })
                            all_paths.append(base)

    # Final dataframe
    final_df = pd.DataFrame(all_paths)

    # Column order
    final_columns = [
        'Source Full Name', 'Target Full Name', 'Ultimate Root Source', 'Final Target',
        'Lineage Path', 'Hop Count', 'Node Level', 'Is Leaf Node',
        'Source DB Name', 'Source Schema Name', 'SourceTableName',
        'Target DB Name', 'Target Schema Name', 'Target Table Name'
    ]
    final_df = final_df[final_columns]

    # Save result
    input_file_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"Result_{input_file_name}.xlsx"
    final_df.to_excel(output_path, index=False)
    print(f"✅ Lineage output saved as: {output_path}")

if __name__ == "__main__":
    Tk().withdraw()
    input_file = filedialog.askopenfilename(
        title="Select your lineage Excel file",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    if input_file:
        generate_lineage_output(input_file)
    else:
        print("❌ No input file selected.")
