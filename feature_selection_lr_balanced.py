import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import PowerTransformer

import pickle

training_data_full= pd.read_parquet('data/parquet_files/training_data_rfm_balanced.parquet')

lr = LogisticRegression(max_iter=1000)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(training_data_full.drop(columns=['Flag']), training_data_full['Flag'], test_size=0.2, random_state=42)

#scale data
pt = PowerTransformer()
X_train_full_scaled = pt.fit_transform(X_train_full)
X_test_full_scaled = pt.transform(X_test_full)

if not os.path.exists('models'):
    os.makedirs('models')

# Sequential Feature Selection with Logistic Regression
for num_features in range(10, 16):
    sfs_lr = SequentialFeatureSelector(lr, n_features_to_select=num_features)
    sfs_lr.fit(X_train_full_scaled, X_test_full_scaled)
    sfs_feat_lr = X_train_full.columns[sfs_lr.get_support()]

    print("No. of Selected Features by Logistic Regression ({} features): {}".format(num_features, len(sfs_feat_lr)))
    print("Logistic Regression Selected Features: ", sfs_feat_lr)

    with open('models/sfs_lr_{}_balanced.pkl'.format(num_features), 'wb') as f:
        pickle.dump(sfs_lr, f)
