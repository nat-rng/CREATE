import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split

import pickle

training_scam_data = pd.read_parquet('data/parquet_files/training_scam_data.parquet')
training_scam_data.drop(columns=['address_id', 'year_month'], inplace=True)

fill_values = {'median_recency_out':0, 'median_recency_in':0, 'num_outliers_eth_out':0, 'num_outliers_eth_in': 0,
               'daily_from_gini_index': 1000, 'daily_to_gini_index': 1000, 'weekly_from_gini_index': 1000, 
               'weekly_to_gini_index': 1000, 'daily_total_gini_index': 1000, 'weekly_total_gini_index': 1000}
training_data_full= training_scam_data.fillna(fill_values)

xgb = XGBClassifier(n_jobs=-1)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(training_data_full.drop(columns=['Flag']), training_data_full['Flag'], test_size=0.2, random_state=42)

if not os.path.exists('models'):
    os.makedirs('models')

ten_fold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

X_train_sfs_xgb = pd.read_pickle('models/X_train_sfs_xgb.pkl')

params = {
    'eta': [0.2, 0.3, 0.4],
    'min_child_weight': [1, 5, 10], 
    'max_depth': [5, 6, 7],
    'gamma': [0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'scale_pos_weight': [1, 2, 3],
    'reg_alpha': [0.5, 1],
    'reg_lambda': [1, 1.5], 
    'random_state': [42] 
}

grid_xgb = GridSearchCV(xgb, param_grid=params, cv=ten_fold, scoring='f1',
                        return_train_score=True, n_jobs=-1)
grid_xgb.fit(X_train_sfs_xgb, y_train_full)

print("Best Parameters {}".format(grid_xgb.best_params_))
print("Best Score {}".format(grid_xgb.best_score_))

with open('models/grid_xgb_cpu.pkl', 'wb') as f:
    pickle.dump(grid_xgb, f)
