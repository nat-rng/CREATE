import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import community as community_louvain

if not os.path.exists('data/graph_files'):
    os.makedirs('data/graph_files')

potential_fraud_df = pd.read_parquet('data/parquet_files/potential_fraud_transactions_df_bal.parquet')
potential_fraud_df = potential_fraud_df[potential_fraud_df['to_id'].isnull()==False]
potential_fraud_df['to_id'] = potential_fraud_df['to_id'].astype('int64')
potential_fraud_df = potential_fraud_df.fillna(value={'asset_value': 0})
potential_fraud_df = potential_fraud_df[potential_fraud_df['asset'].isnull()==False]
#encode asset column
potential_fraud_df['asset'] = potential_fraud_df['asset'].astype('category')
potential_fraud_df = potential_fraud_df[['from_id', 'to_id', 'asset_value', 'asset', 'category_id']]
potential_fraud_eth_df = potential_fraud_df[(potential_fraud_df['asset']=='ETH') & (potential_fraud_df['category_id']!=3)]
potential_fraud_eth_df = potential_fraud_eth_df.fillna(value={'asset_value': 0})
potential_fraud_eth_df['asset_value'] = potential_fraud_eth_df['asset_value'].astype('float')
agg_eth_df = potential_fraud_eth_df.groupby(['from_id', 'to_id'], as_index=False).agg({'asset_value': 'sum'})

G = nx.from_pandas_edgelist(agg_eth_df, 'from_id', 'to_id', edge_attr='asset_value', create_using=nx.DiGraph())

#compute pagerank, betweenness centrality, degree centrality
pr = nx.pagerank(G, alpha=0.9)
bc = nx.betweenness_centrality(G)
dc = nx.degree_centrality(G)

#create dataframe with pagerank, betweenness centrality, degree centrality for each node
data = {'pagerank': pr, 'betweenness_centrality': bc, 'degree_centrality': dc}
centrality_df = pd.DataFrame(data).reset_index().rename(columns={'index': 'node_id'})

partition = community_louvain.best_partition(G)
data = {'node_id': list(partition.keys()), 'community': list(partition.values())}
community_df = pd.DataFrame(data)
community_df.to_parquet('data/graph_files/eth_community_df_bal.parquet')

# Create a color map, one for each partition
color_map = cm.get_cmap('nipy_spectral', max(partition.values()) + 1)

# Apply community to graph and color nodes by their community
for node in G.nodes():
    G.nodes[node]['community'] = partition[node]
    G.nodes[node]['color'] = color_map(partition[node])

node_colors = [G.nodes[node]['color'] for node in G.nodes()]

plt.figure(figsize=(1000, 720))
layout = nx.kamada_kawai_layout(G)

nx.draw(G, layout, node_size=2, node_color=node_colors, width=0.1, alpha=0.3, with_labels=False)

plt.savefig("data/graph_files/eth_community_graph_bal.png")
plt.close()
