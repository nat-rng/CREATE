from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold
import pandas as pd
import os
import pickle
import multiprocessing

training_data_full= pd.read_parquet('data/parquet_files/training_data_rfm.parquet')

_, _, y_train_full, _ = train_test_split(training_data_full.drop(columns=['Flag']), training_data_full['Flag'], test_size=0.2, random_state=42)
X_train_sfs_xgb = pd.read_pickle('models/X_train_sfs_xgb.pkl')
ten_fold = RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=5)

def train_xgboost(config, checkpoint_dir=None):
    clf = XGBClassifier(n_jobs=-1, **config)
    scores = cross_val_score(clf, X_train_sfs_xgb, y_train_full, cv=ten_fold, scoring='f1')
    f1 = scores.mean()
    tune.report(loss=1-f1)

config = {
    'eta': tune.loguniform(0.01, 1.0),
    'max_depth': tune.randint(1, 15),
    'min_child_weight': tune.randint(1, 11),
    'gamma': tune.uniform(0, 1),
    'lambda': tune.uniform(0, 2),
    'alpha': tune.uniform(0, 1),
    'scale_pos_weight': tune.uniform(0.1, 10),
    'subsample': tune.uniform(0.5, 1.0),
    'colsample_bytree': tune.uniform(0.5, 1.0),
    'n_estimators': tune.randint(50, 500)
}

bohb_hyperband = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=350, 
    reduction_factor=3.5,
    metric="loss",
    mode="min")

bohb_search = TuneBOHB(
    max_concurrent=multiprocessing.cpu_count(),
    metric="loss",
    mode="min")

analysis = tune.run(
    train_xgboost,
    config=config,
    scheduler=bohb_hyperband,
    search_alg=bohb_search,
    num_samples=100
)

best_trial = analysis.get_best_trial("loss", "min", "last")
print('Best found configuration:', best_trial.config)

bohb_best_params = best_trial.config
best_loss = best_trial.last_result['loss']

best_trial_dict = {'best_params': bohb_best_params, 'best_loss': best_loss}

if not os.path.exists('models'):
    os.makedirs('models')

with open("models/bohb_xgb.pkl", "wb") as f:
    pickle.dump(best_trial_dict, f)