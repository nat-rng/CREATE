import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, train_test_split
import pickle

os.environ['KMP_DUPLICATE_LIB_OK']='True'

training_data_full= pd.read_parquet('data/parquet_files/training_data_rfm_balanced.parquet')

xgb = XGBClassifier(n_jobs=-1)

_, _, y_train_full, _ = train_test_split(training_data_full.drop(columns=['Flag']), training_data_full['Flag'], test_size=0.2, random_state=42)

if not os.path.exists('models'):
    os.makedirs('models')

ten_fold = RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=5)

X_train_sfs_xgb = pd.read_pickle('models/X_train_sfs_xgb_balanced.pkl')

params = {
    'n_estimators': [10, 50, 100, 200, 300], 
    'eta': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], 
    'min_child_weight': [1, 5, 10],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 2, 3, 4],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [1, 1.5, 2], 
    'random_state': [42] 
}

randomized_xgb = RandomizedSearchCV(xgb, param_distributions=params, cv=ten_fold, scoring='f1', 
                                    return_train_score=True, n_jobs=-1, n_iter=8000, random_state=42)
randomized_xgb.fit(X_train_sfs_xgb, y_train_full)

print("Best Parameters {}".format(randomized_xgb.best_params_))
print("Best Score {}".format(randomized_xgb.best_score_))

with open('models/randomized_xgb_balanced.pkl', 'wb') as f:
    pickle.dump(randomized_xgb, f)