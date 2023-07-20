import pandas as pd
import networkx as nx
import os

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
centrality_df.to_parquet('data/graph_files/eth_centrality_df_bal.parquet')