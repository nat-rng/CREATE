import pandas as pd
from sklearn.model_selection import train_test_split
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier
from sklearn.feature_selection import SequentialFeatureSelector

import pickle

training_data_full= pd.read_parquet('data/parquet_files/training_data_rfm.parquet')

xgb = XGBClassifier(n_jobs=-1)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(training_data_full.drop(columns=['Flag']), training_data_full['Flag'], test_size=0.2, random_state=42)

if not os.path.exists('models'):
    os.makedirs('models')

# Sequential Feature Selection with XGBoost
for num_features in range(10, 16):
    sfs_xgb = SequentialFeatureSelector(xgb, n_features_to_select=num_features)
    sfs_xgb.fit(X_train_full, y_train_full)
    sfs_feat_xgb = X_train_full.columns[sfs_xgb.get_support()]

    print("No. of Selected Features by XGBoost ({} features): {}".format(num_features, len(sfs_feat_xgb)))
    print("XGBoost Selected Features: ", sfs_feat_xgb)

    with open('models/sfs_xgb_{}.pkl'.format(num_features), 'wb') as f:
        pickle.dump(sfs_xgb, f)
