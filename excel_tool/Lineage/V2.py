import pandas as pd
import networkx as nx
import os

def generate_lineage_from_excel(input_file: str) -> pd.DataFrame:
    """
    Generates lineage details from an input Excel file and writes output with 'Result_' prefix.

    Args:
        input_file (str): Path to the input Excel file.

    Returns:
        pd.DataFrame: Final enriched lineage DataFrame.
    """
    # === Read Excel ===
    df = pd.read_excel(input_file)

    # === Build full names ===
    df['Source Full Name'] = df['Source DB Name'] + '.' + df['Source Schema Name'] + '.' + df['SourceTableName']
    df['Target Full Name'] = df['Target DB Name'] + '.' + df['Target Schema Name'] + '.' + df['Target Table Name']

    # === Build graph ===
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Source Full Name'], row['Target Full Name'])
    G_reverse = G.reverse()

    def find_root_source(node):
        while True:
            preds = list(G_reverse.predecessors(node))
            if not preds:
                return node
            node = preds[0]

    def find_paths(graph, source):
        paths = []
        for node in nx.descendants(graph, source):
            if graph.out_degree(node) == 0:
                for path in nx.all_simple_paths(graph, source=source, target=node):
                    paths.append(path)
        return paths

    # === Build lineage records ===
    lineage_records = []
    for source in df['Source Full Name'].unique():
        paths = find_paths(G, source)
        if not paths:
            for target in df[df['Source Full Name'] == source]['Target Full Name'].unique():
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

    # === Merge with original and enrich ===
    merged_df = pd.merge(df, lineage_df, on=['Source Full Name', 'Target Full Name'], how='left')
    merged_df['Ultimate Root Source'] = merged_df['Source Full Name'].apply(find_root_source)
    merged_df['Is Leaf Node'] = merged_df['Target Full Name'].apply(lambda x: G.out_degree(x) == 0)

    # === Reorder columns: lineage first
    lineage_cols = ['Ultimate Root Source', 'Final Target', 'Lineage Path', 'Hop Count', 'Is Leaf Node']
    all_cols = lineage_cols + [col for col in merged_df.columns if col not in lineage_cols]
    final_df = merged_df[all_cols]

    # === Save to output file ===
    filename = os.path.basename(input_file)
    base, ext = os.path.splitext(filename)
    output_file = os.path.join(os.path.dirname(input_file), f"Result_{base}.xlsx")
    final_df.to_excel(output_file, index=False)

    print(f"\n✅ Lineage file saved as: {output_file}")
    return final_df


# === MAIN CALL ===
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    # === File picker GUI ===
    root = tk.Tk()
    root.withdraw()
    input_excel_path = filedialog.askopenfilename(
        title="Select your lineage Excel file",
        filetypes=[("Excel Files", "*.xlsx *.xls")]
    )

    if not input_excel_path:
        print("❌ No file selected. Exiting.")
    else:
        df_lineage = generate_lineage_from_excel(input_excel_path)
        print("\n🔍 Sample Output:")
        print(df_lineage.head())
