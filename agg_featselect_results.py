import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


training_data_rfm = pd.read_parquet('data/parquet_files/training_data_rfm.parquet')
X_train, X_test, y_train, y_test = train_test_split(training_data_rfm.drop(columns=['Flag']), training_data_rfm['Flag'], test_size=0.2, random_state=42)
pt = PowerTransformer()
X_train_scaled = pt.fit_transform(X_train)

scoring = {'accuracy': make_scorer(accuracy_score), 
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score)}

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

results = []

for i in range(10, 16):
    for model_name in ["lr", "rf", "xgb"]:
        model_file = f"models/sfs_{model_name}_{i}.pkl"
        sfs_model = pd.read_pickle(model_file)

        sfs_feat_xgb= X_train.columns[(sfs_model.get_support())]
        num_features = len(sfs_feat_xgb)
        selected_features = list(sfs_feat_xgb)

        X_train_sub, y_train_sub = sfs_model.transform(X_train), y_train
        if model_name == "lr":
            smote = SMOTE(random_state=42)
            X_train_sub, y_train_sub = smote.fit_resample(sfs_model.transform(X_train_scaled), y_train)
            estimator = LogisticRegression(max_iter=1000)
        elif model_name == "rf":
            estimator = RandomForestClassifier(n_jobs=-1)
        else:
            estimator = XGBClassifier(n_jobs=-1)

        scores = cross_validate(estimator, X_train_sub, y_train_sub, cv=rskf, scoring=scoring)

        result = {"model": f"{model_name}_{i}", 
                  "num_features": num_features,
                  "selected_features": selected_features,
                  "accuracy": scores["test_accuracy"].mean(),
                  "precision": scores["test_precision"].mean(),
                  "recall": scores["test_recall"].mean(),
                  "f1": scores["test_f1"].mean()}
        results.append(result)

results_df = pd.DataFrame(results)

results_df.to_parquet("data/parquet_files/agg_featselect_results.parquet")