from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import pickle

X_train_sfs_xgb_balanced = pd.read_pickle('models/X_train_sfs_xgb_balanced.pkl')
chosen_features = X_train_sfs_xgb_balanced.columns

all_june_accounts_df = pd.read_parquet('data/parquet_files/all_june_accounts_df.parquet')

xgb_models = pd.read_pickle('models/xgb_bohb_balanced_models.pkl')
X_test = all_june_accounts_df[chosen_features]

all_predictions = []
for model in xgb_models:
    proba = model.predict_proba(X_test)
    all_predictions.append(proba)

all_predictions = np.array(all_predictions)
normal_lb = np.percentile(all_predictions[:,:,0], 2.5, axis=0)
normal_ub = np.percentile(all_predictions[:,:,0], 97.5, axis=0)

fraud_lb = np.percentile(all_predictions[:,:,1], 2.5, axis=0)
fraud_ub = np.percentile(all_predictions[:,:,1], 97.5, axis=0)

all_june_accounts_df['normal_ci'] = list(zip(normal_lb, normal_ub))
all_june_accounts_df['fraud_ci'] = list(zip(fraud_lb, fraud_ub))

confidence_interval_df = all_june_accounts_df[['address_id', 'normal_ci', 'fraud_ci']]

with open('data/parquet_files/confidence_interval_df.pkl', 'wb') as f:
    pickle.dump(confidence_interval_df, f)

