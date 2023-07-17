import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.feature_selection import SequentialFeatureSelector

import pickle

training_data_full= pd.read_parquet('data/parquet_files/training_data_rfm_balanced.parquet')

rf = RandomForestClassifier(n_jobs=-1)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(training_data_full.drop(columns=['Flag']), training_data_full['Flag'], test_size=0.2, random_state=42)

if not os.path.exists('models'):
    os.makedirs('models')

# Sequential Feature Selection with Random Forest
for num_features in range(10, 16):
    sfs_rf = SequentialFeatureSelector(rf, n_features_to_select=num_features)
    sfs_rf.fit(X_train_full, y_train_full)
    sfs_feat_rf = X_train_full.columns[sfs_rf.get_support()]

    print("No. of Selected Features by Random Forest ({} features): {}".format(num_features, len(sfs_feat_rf)))
    print("Random Forest Selected Features: ", sfs_feat_rf)

    with open('models/sfs_rf_{}_balanced.pkl'.format(num_features), 'wb') as f:
        pickle.dump(sfs_rf, f)
