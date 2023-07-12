import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import logging
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold
from xgboost import XGBClassifier
import multiprocessing

import os
import pickle
import pandas as pd

training_data_full= pd.read_parquet('data/parquet_files/training_data_rfm.parquet')

_, _, y_train_full, _ = train_test_split(training_data_full.drop(columns=['Flag']), training_data_full['Flag'], test_size=0.2, random_state=42)
X_train_sfs_xgb = pd.read_pickle('models/X_train_sfs_xgb.pkl')
ten_fold = RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=5)

class XGBoostWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train = X_train_sfs_xgb
        self.y_train = y_train_full

    def compute(self, config, budget,  working_directory, *args, **kwargs):
        config['min_child_weight'] = int(budget)
        clf = XGBClassifier(**config, n_jobs=-1)
        scores = cross_val_score(clf, self.x_train, self.y_train, cv=ten_fold, scoring='f1')
        acc = scores.mean()  

        return ({
            'loss': 1-acc,  
            'info': {}
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('eta', lower=0.01, upper=0.5))
        config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('n_estimators', lower=50, upper=300))
        config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('max_depth', lower=3, upper=8))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('gamma', lower=0.0, upper=0.4))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('subsample', lower=0.5, upper=1.0))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('colsample_bytree', lower=0.6, upper=1.0))
        config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('scale_pos_weight', lower=1, upper=4))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('reg_alpha', lower=0.0, upper=1.0))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('reg_lambda', lower=1.0, upper=2.0))

        return config_space

NS = hpns.NameServer(run_id='xgb_run', host='localhost', port=None)
NS.start()

num_cores = multiprocessing.cpu_count()
# start multiple instances of the worker
workers = []
for i in range(num_cores):  # adjust the number according to your available cores
    w = XGBoostWorker(nameserver='localhost', run_id='xgb_run', id='xgbworker_{}'.format(i))
    w.run(background=True)
    workers.append(w)

bohb = BOHB(configspace=w.get_configspace(),
            run_id='xgb_run', nameserver='localhost',
            min_budget=1, max_budget=10)

res = bohb.run(n_iterations=16, min_n_workers=num_cores)

bohb.shutdown(shutdown_workers=True)
NS.shutdown()

id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])

if not os.path.exists('models'):
    os.makedirs('models')

with (open("models/bohb_xgb.pkl", "wb")) as f:
    pickle.dump(res, f)