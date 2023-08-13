import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import PowerTransformer

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SVMSMOTE
from sklearn.neighbors import NearestNeighbors

training_data_rfm = pd.read_parquet('data/parquet_files/training_data_rfm.parquet')

X_train, X_test, y_train, y_test = train_test_split(training_data_rfm.drop(columns=['Flag']), training_data_rfm['Flag'], test_size=0.2, random_state=42)
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

scoring = {'accuracy': make_scorer(accuracy_score), 
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score)}

five_neighbours = NearestNeighbors(n_jobs=-1)
svmsmote = SVMSMOTE(k_neighbors=five_neighbours, random_state=42)
smote_lr_pipeline = make_pipeline(svmsmote, LogisticRegression(max_iter=1000))
smote_rf_pipeline = make_pipeline(svmsmote, RandomForestClassifier())
smote_xgb_pipeline = make_pipeline(svmsmote, XGBClassifier())

pt = PowerTransformer()
X_train_scaled = pt.fit_transform(X_train)

scores_svmsmote_lr = cross_validate(smote_lr_pipeline, X_train_scaled, y_train, cv=rskf, scoring=scoring)
print(f'LR Accuracy: {scores_svmsmote_lr["test_accuracy"].mean()}')
print(f'LR Precision: {scores_svmsmote_lr["test_precision"].mean()}')
print(f'LR Recall: {scores_svmsmote_lr["test_recall"].mean()}')
print(f'LR F1: {scores_svmsmote_lr["test_f1"].mean()}')

scores_svmsmote_rf = cross_validate(smote_rf_pipeline, X_train, y_train, cv=rskf, scoring=scoring)
print(f'RF Accuracy: {scores_svmsmote_rf["test_accuracy"].mean()}')
print(f'RF Precision: {scores_svmsmote_rf["test_precision"].mean()}')
print(f'RF Recall: {scores_svmsmote_rf["test_recall"].mean()}')
print(f'RF F1: {scores_svmsmote_rf["test_f1"].mean()}')

scores_svmsmote_xgb = cross_validate(smote_xgb_pipeline, X_train, y_train, cv=rskf, scoring=scoring)
print(f'XGB Accuracy: {scores_svmsmote_xgb["test_accuracy"].mean()}')
print(f'XGB Precision: {scores_svmsmote_xgb["test_precision"].mean()}')
print(f'XGB Recall: {scores_svmsmote_xgb["test_recall"].mean()}')
print(f'XGB F1: {scores_svmsmote_xgb["test_f1"].mean()}')

with open('data/pickle_files/scores_svmsmote_lr.pkl', 'wb') as f:
    pickle.dump(scores_svmsmote_lr, f)

with open('data/pickle_files/scores_svmsmote_rf.pkl', 'wb') as f:
    pickle.dump(scores_svmsmote_rf, f)

with open('data/pickle_files/scores_svmsmote_xgb.pkl', 'wb') as f:
    pickle.dump(scores_svmsmote_xgb, f)