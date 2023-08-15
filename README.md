# CREATE
---

## Description
This repository contains the collection of Python scripts used for, validating ethereum addresse, feature selection, model validation, and graph generation. Each file serves a specific purpose as detailed below.

## Python Version
* Python 3.8.X, 3.9.X, 3.11.X

## Libraries Used
* pandas
* numpy
* pickle
* fastparquet
* xgboost
* sklearn
* imblearn
* networkx
* python-louvain
* ray-tune
* hpbandster
* configspace
* web3


## File Descriptions

- **agg_featselect_results_balanced.py**: 
    - Description: Aggregates results from balanced feature selection for different number of features.

- **agg_featselect_results.py**: 
    - Description: Aggregates results from feature selection for different number of features.

- **alchemy_api.py**: 
    - Description: File related to the Alchemy API functionalities and interactions.

- **ci_models_balanced.py**: 
    - Description: Script dedicated to confidence interval computations for balanced models.

- **ci_models.py**: 
    - Description: Script for computing confidence intervals for RFM models.

- **compute_ci.py**: 
    - Description: Script for computing confidence intervals for predictions of the final model.

- **create_community_graph.py**: 
    - Description: Script for creating community graphs.

- **create_graph_files.py**: 
    - Description: Script to process transaction data into a DAG.

- **create_graph_g.py**: 
    - Description: Script to process transaction data into a DAG and outputting it as a graph file.

- **create_subcommunity_graph.py**: 
    - Description: Script for creating sub-community graphs.

- **cross_val_bohb_bal.py**: 
    - Description: Cross-validation script using BOHB for balanced data.

- **cross_val_bohb.py**: 
    - Description: Cross-validation script using BOHB.

- **cross_validation_cpu.py**: 
    - Description: Cross-validation script optimized using GridSearch.

- **cross_validation_rand_cpu_balanced.py**: 
    - Description: Balanced cross-validation script optimized for randomsearch.

- **cross_validation_rand_cpu.py**: 
    - Description: Cross-validation script optimized for random.

- **feature_selection_lr_balanced.py**: 
    - Description: Feature selection script using logistic regression for balanced data.

- **feature_selection_lr.py**: 
    - Description: Feature selection using logistic regression.

- **feature_selection_rf_balanced.py**: 
    - Description: Feature selection script using random forest for balanced data.

- **feature_selection_rf.py**: 
    - Description: Feature selection using random forest.

- **feature_selection_xgb_balanced.py**: 
    - Description: Feature selection script using XGBoost for balanced data.

- **feature_selection_xgb.py**: 
    - Description: Feature selection using XGBoost.

- **svm_smote.py**: 
    - Description: Script for testing SVM SMOTE (Synthetic Minority Over-sampling Technique) performance on all models.

- **verify_contracts_balanced.py**: 
    - Description: Script to verify contracts of balanced data.

- **verify_contracts_may_june_balanced.py**: 
    - Description: Script for verifying contracts from May to June using balanced data.

- **verify_contracts_may_june.py**: 
    - Description: Script to verify contracts from May to June.

- **verify_contracts.py**: 
    - Description: Script to verify all contracts in June.

## Installation and Setup
1. Clone the repository.
2. Download data files at <https://emckclac-my.sharepoint.com/:f:/g/personal/k2257934_kcl_ac_uk/EvtzfQMAtfdMtc-6AwZGfLIBPYqwBFQiEgxBQ0o_2yAJFw?e=ssmHcu>
3. Install the required libraries mentioned above and any additional ones if errors are thrown.
    conda instal <library-name>
4. Execute individual scripts as per requirements.