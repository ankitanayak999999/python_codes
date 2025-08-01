import pandas as pd
import networkx as nx

# Step 1: Read your Excel file (update file path as needed)
input_path = "your_input_file.xlsx"  # Replace this with your actual file path
df = pd.read_excel(input_path)

# Step 2: Create Source and Target Full Names
df['Source Full Name'] = df['Source DB Name'] + '.' + df['Source Schema Name'] + '.' + df['SourceTableName']
df['Target Full Name'] = df['Target DB Name'] + '.' + df['Target Schema Name'] + '.' + df['Target Table Name']

# Step 3: Build lineage graph
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['Source Full Name'], row['Target Full Name'])

# Step 4: Reverse graph to trace ultimate root sources
G_reverse = G.reverse()

def find_root_source(node):
    while True:
        predecessors = list(G_reverse.predecessors(node))
        if not predecessors:
            return node
        node = predecessors[0]

# Step 5: Get all paths from each source
def find_paths(graph, source):
    paths = []
    for node in nx.descendants(graph, source):
        if graph.out_degree(node) == 0:  # leaf node
            for path in nx.all_simple_paths(graph, source=source, target=node):
                paths.append(path)
    return paths

# Step 6: Build lineage path table
all_paths = []
for source in df['Source Full Name'].unique():
    paths = find_paths(G, source)
    if not paths:
        for target in df[df['Source Full Name'] == source]['Target Full Name'].unique():
            all_paths.append({
                'Source Full Name': source,
                'Target Full Name': target,
                'Final Target': target,
                'Lineage Path': f"{source.split('.')[-1]} → {target.split('.')[-1]}",
                'Hop Count': 1
            })
    else:
        for path in paths:
            all_paths.append({
                'Source Full Name': path[0],
                'Target Full Name': path[1] if len(path) > 1 else path[0],
                'Final Target': path[-1],
                'Lineage Path': ' → '.join([p.split('.')[-1] for p in path]),
                'Hop Count': len(path) - 1
            })

lineage_df = pd.DataFrame(all_paths).drop_duplicates()

# Step 7: Merge lineage with original data
final_df = pd.merge(df, lineage_df, on=['Source Full Name', 'Target Full Name'], how='left')

# Step 8: Add true ultimate root and leaf node flag
final_df['Ultimate Root Source'] = final_df['Source Full Name'].apply(find_root_source)
final_df['Is Leaf Node'] = final_df['Target Full Name'].apply(lambda x: G.out_degree(x) == 0)

# Step 9: Save to Excel
output_path = "lineage_output_final.xlsx"
final_df.to_excel(output_path, index=False)

# Optional: Preview
print(final_df[['Source Full Name', 'Target Full Name', 'Ultimate Root Source', 'Final Target', 'Lineage Path', 'Is Leaf Node']])
