import pandas as pd
import networkx as nx
import os

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
nx.write_gexf(G, 'data/graph_files/eth_graph.gexf')