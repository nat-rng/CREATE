import pandas as pd
import networkx as nx
import os
import community as community_louvain
import random
random.seed(42)

if not os.path.exists('data/graph_files'):
    os.makedirs('data/graph_files')

eth_tx_df = pd.read_parquet('data/parquet_files/all_eth_transactions_df.parquet')
eth_tx_df = eth_tx_df[eth_tx_df['to_id'].isnull()==False]
eth_tx_df['to_id'] = eth_tx_df['to_id'].astype('int64')
eth_tx_df = eth_tx_df.fillna(value={'asset_value': 0})
eth_tx_df = eth_tx_df[eth_tx_df['asset'].isnull()==False]
#encode asset column
eth_tx_df['asset'] = eth_tx_df['asset'].astype('category')
eth_tx_df = eth_tx_df[['from_id', 'to_id', 'asset_value', 'asset', 'category_id']]
eth_tx_df = eth_tx_df[(eth_tx_df['asset']=='ETH') & (eth_tx_df['category_id']!=3)]
eth_tx_df = eth_tx_df.fillna(value={'asset_value': 0})
eth_tx_df['asset_value'] = eth_tx_df['asset_value'].astype('float')
agg_eth_df = eth_tx_df.groupby(['from_id', 'to_id'], as_index=False).agg({'asset_value': 'sum'})

G = nx.from_pandas_edgelist(agg_eth_df, 'from_id', 'to_id', edge_attr='asset_value', create_using=nx.DiGraph())

#compute pagerank, degree centrality and weighted degree
pr = nx.pagerank(G, alpha=0.9)
dc = nx.degree_centrality(G)
wd = nx.degree(G, weight='asset_value')
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

pr_series = pd.Series(pr, name='pagerank')
dc_series = pd.Series(dc, name='degree_centrality')
wd_series = pd.Series(wd, name='weighted_degree')
in_degree_series = pd.Series(in_degree, name='in_degree')
out_degree_series = pd.Series(out_degree, name='out_degree')

centrality_df = pd.concat([pr_series, dc_series, wd_series, in_degree_series, out_degree_series], axis=1).reset_index().rename(columns={'index': 'node_id'})

components = nx.weakly_connected_components(G)
largest_component = max(components, key=len)
sub_graph = G.subgraph(largest_component)

components = nx.weakly_connected_components(G)
node_to_comp = {node: i for i, comp in enumerate(components) for node in comp}
component_df = pd.DataFrame.from_dict(node_to_comp, orient='index', columns=['component_id']).reset_index().rename(columns={'index': 'node_id'})
#convert to undirected graph
sub_graph = sub_graph.to_undirected()
partition = community_louvain.best_partition(sub_graph, random_state=42)

partition_df = pd.DataFrame.from_dict(partition, orient='index', columns=['community_id']).reset_index().rename(columns={'index': 'node_id'})

centrality_df = centrality_df.merge(component_df, on='node_id', how='left')
centrality_df = centrality_df.merge(partition_df, on='node_id', how='left')

centrality_df.to_parquet('data/graph_files/eth_centrality_df.parquet')
component_df.to_parquet('data/graph_files/eth_component_df.parquet')
partition_df.to_parquet('data/graph_files/eth_partition_df.parquet')