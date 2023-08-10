import pandas as pd
import networkx as nx
import os

if not os.path.exists('data/graph_files'):
    os.makedirs('data/graph_files')

partition_df = pd.read_parquet('data/graph_files/eth_partition_df.parquet')
print(partition_df[partition_df['community_id'] == 1]['node_id'].tolist())
G = nx.read_gexf('data/graph_files/eth_graph.gexf')
print(len(G.nodes()))
community_one = G.subgraph(partition_df[partition_df['community_id'] == 1]['node_id'].tolist())
print(len(community_one.nodes()))
nx.write_graphml(community_one, 'data/graph_files/eth_community_one.graphml')