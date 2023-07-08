import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import PowerTransformer

from imblearn.over_sampling import ADASYN

import pickle

training_scam_data = pd.read_parquet('data/parquet_files/training_scam_data.parquet')
training_scam_data.drop(columns=['address_id', 'year_month'], inplace=True)

fill_values = {'median_recency_out':0, 'median_recency_in':0, 'num_outliers_eth_out':0, 'num_outliers_eth_in': 0,
               'daily_from_gini_index': 1000, 'daily_to_gini_index': 1000, 'weekly_from_gini_index': 1000, 
               'weekly_to_gini_index': 1000, 'daily_total_gini_index': 1000, 'weekly_total_gini_index': 1000}
training_data_full= training_scam_data.fillna(fill_values)

lr = LogisticRegression(max_iter=1000)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(training_data_full.drop(columns=['Flag']), training_data_full['Flag'], test_size=0.2, random_state=42)

#scale data
pt = PowerTransformer()
X_train_full_scaled = pt.fit_transform(X_train_full)
X_test_full_scaled = pt.transform(X_test_full)

if not os.path.exists('models'):
    os.makedirs('models')

adasyn = ADASYN(random_state=42)
X_train_full_adasyn, y_train_full_adasyn = adasyn.fit_resample(X_train_full_scaled, y_train_full)
# Sequential Feature Selection with Logistic Regression
for num_features in range(11, 16):
    sfs_lr = SequentialFeatureSelector(lr, n_features_to_select=num_features)
    sfs_lr.fit(X_train_full_adasyn, y_train_full_adasyn)
    sfs_feat_lr = X_train_full.columns[sfs_lr.get_support()]

    print("No. of Selected Features by Logistic Regression ({} features): {}".format(num_features, len(sfs_feat_lr)))
    print("Logistic Regression Selected Features: ", sfs_feat_lr)

    with open('models/sfs_lr_{}.pkl'.format(num_features), 'wb') as f:
        pickle.dump(sfs_lr, f)
