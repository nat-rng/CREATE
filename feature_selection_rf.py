import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.feature_selection import SequentialFeatureSelector

import pickle

training_scam_data = pd.read_parquet('data/parquet_files/training_scam_data.parquet')
training_scam_data.drop(columns=['address_id', 'year_month'], inplace=True)

fill_values = {'median_recency_out':0, 'median_recency_in':0, 'num_outliers_eth_out':0, 'num_outliers_eth_in': 0,
               'daily_from_gini_index': 1000, 'daily_to_gini_index': 1000, 'weekly_from_gini_index': 1000, 
               'weekly_to_gini_index': 1000, 'daily_total_gini_index': 1000, 'weekly_total_gini_index': 1000}
training_data_full= training_scam_data.fillna(fill_values)

rf = RandomForestClassifier(n_jobs=-1)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(training_data_full.drop(columns=['Flag']), training_data_full['Flag'], test_size=0.2, random_state=42)

if not os.path.exists('models'):
    os.makedirs('models')

# Sequential Feature Selection with Random Forest
for num_features in range(11, 16):
    sfs_rf = SequentialFeatureSelector(rf, n_features_to_select=num_features)
    sfs_rf.fit(X_train_full, y_train_full)
    sfs_feat_rf = X_train_full.columns[sfs_rf.get_support()]

    print("No. of Selected Features by Random Forest ({} features): {}".format(num_features, len(sfs_feat_rf)))
    print("Random Forest Selected Features: ", sfs_feat_rf)

    with open('models/sfs_rf_{}.pkl'.format(num_features), 'wb') as f:
        pickle.dump(sfs_rf, f)
