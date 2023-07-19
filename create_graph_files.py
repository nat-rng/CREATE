import pandas as pd
import networkx as nx
import os

if not os.path.exists('data/gexf_files'):
    os.makedirs('data/gexf_files')

def create_and_save_graph(input_file_path, output_file_path):
    df = pd.read_parquet(input_file_path)
    df = df[df['to_id'].isnull() == False]
    df['to_id'] = df['to_id'].astype('int64')
    df = df.fillna(value={'asset_value': 0})
    df = df[df['asset'].isnull() == False]

    # encode asset column
    df['asset'] = df['asset'].astype('category')

    # create a directed graph
    G = nx.from_pandas_edgelist(df, 'from_id', 'to_id', ['asset_value', 'asset', 'category_id'], create_using=nx.DiGraph())
    
    # write to a gexf file
    nx.write_gexf(G, output_file_path)

# Using the function for both datasets
create_and_save_graph('data/parquet_files/potential_fraud_transactions_df_bal.parquet', 'data/gexf_files/potential_fraud_bal.gexf')
create_and_save_graph('data/parquet_files/potential_fraud_transactions_df.parquet', 'data/gexf_files/potential_fraud.gexf')
