import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
import pickle

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

ten_fold = KFold(n_splits=10, random_state=42, shuffle=True)

X_train_sfs_xgb = pd.read_pickle('models/X_train_sfs_xgb.pkl')

params = {
    'eta': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], # learning rate
    'min_child_weight': [1, 5, 10], # minimum sum of instance weight (hessian) needed in a child
    'max_depth': [3, 4, 5, 6, 7, 8], # maximum depth of a tree
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4], # minimum loss reduction required to make a split
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # fraction of observations to be randomly samples for each tree.
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], # fraction of columns to be randomly samples for each tree.
    'scale_pos_weight': [1, 2, 3, 4], # Control the balance of positive and negative weights
    'reg_alpha': [0, 0.5, 1], # L1 regularization term on weight (analogous to Lasso regression)
    'reg_lambda': [1, 1.5, 2], # L2 regularization term on weights (analogous to Ridge regression)
    'random_state': [42] # seed used to generate reproducible results
}

# Use RandomizedSearchCV instead of GridSearchCV
randomized_xgb = RandomizedSearchCV(xgb, param_distributions=params, cv=ten_fold, scoring='f1', 
                                    return_train_score=True, n_jobs=-1, n_iter=1000, random_state=42)
randomized_xgb.fit(X_train_sfs_xgb, y_train_full)

print("Best Parameters {}".format(randomized_xgb.best_params_))
print("Best Score {}".format(randomized_xgb.best_score_))

with open('models/randomized_xgb_cpu.pkl', 'wb') as f:
    pickle.dump(randomized_xgb, f)