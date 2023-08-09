from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
import pandas as pd
import pickle

training_data_full= pd.read_parquet('data/parquet_files/training_data_rfm_balanced.parquet')
_, _, y_train_full, _ = train_test_split(training_data_full.drop(columns=['Flag']), training_data_full['Flag'], test_size=0.2, random_state=42)
X_train_sfs_xgb = pd.read_pickle('models/X_train_sfs_xgb_balanced.pkl')

best_trial = pd.read_pickle('models/bohb_xgb_balanced.pkl')
bohb_best_params = best_trial['best_params']

bagging_samples = 500

xgb_bohb_models = []

for b in range(bagging_samples):
    xgb_bohb_best = XGBClassifier(**bohb_best_params, n_jobs=-1, random_state=b)
    xgb_bohb_best.fit(X_train_sfs_xgb, y_train_full)
    xgb_bohb_models.append(xgb_bohb_best)

with open('models/xgb_bohb_balanced_models.pkl', 'wb') as f:
    pickle.dump(xgb_bohb_models, f)