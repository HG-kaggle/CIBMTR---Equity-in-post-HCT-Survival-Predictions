# All the group members have been added to the project
import random
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib as mp
import scipy as sp
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import optuna


# Part 1: Data cleaning, adding -1 for numerical missing values, and "NA" string value
# for categorical missing values.

train = pd.read_csv('train.csv')
# This is for test purpose to fix

categorical_columns = train.select_dtypes(exclude=[np.number]).columns
numerical_columns = train.select_dtypes(include=[np.number]).columns

# Linear interpolation for numerical columns in data: we transform all NA into -1.
for col in numerical_columns:
    if train[col].isna().sum() > 0:
        train[col] = train[col].fillna(-1)

# Fill categorical columns with string "NA" value.

for col in categorical_columns:
    if train[col].isna().sum() > 0:
        train[col] = train[col].fillna('NA')

# Check if the processed train dataset has any missing values.
print(train.isnull().sum())

# This NA imputation for NA is not sufficient for categorical variables because
# some categorical variables, such as dri_score, has values of 'N/A - non-malignant indication.'
# These are actually NA and thus needs to be converted into "NA."

# Replace NA-like strings for dri_score:
na_mapping_dri = {
    'Missing disease status': 'NA-1',
    'N/A - disease not classifiable': 'NA-2',
    'N/A - non-malignant indication': 'NA-3',
    'N/A - pediatric': 'NA-4'
}
train['dri_score'] = train['dri_score'].replace(na_mapping_dri)

# Replace NA strings for cyto_score:
na_mapping_cyto = {
    'TBD': 'NA',
    'Other': 'NA',
    'Not tested': 'NA'
}
train['cyto_score'] = train['cyto_score'].replace(na_mapping_cyto)

# Replace NA string for conditioning_intensity:
na_mapping_ci = {
    'N/A, F(pre-TED) not submitted': 'NA',
    'No drugs reported': 'NA'
}
train['conditioning_intensity'] = train['conditioning_intensity'].replace(na_mapping_ci)

# Impute not used value string for melphalan_dose:
na_mapping_mel = {
    "N/A, Mel not given": 'Not Used'
}
train['melphalan_dose'] = train['melphalan_dose'].replace(na_mapping_mel)


def not_done_cleaning(data: pd.DataFrame):
    """
    To automatically replace all "Not done" string values within categorical columns with 'NA'.

    Returns
    -------
    None
    """
    na_mapping = {
        'Not done': 'NA'
    }
    categorical_col = data.select_dtypes(include=['object', 'category']).columns
    # Apply replacement to each categorical column
    for col in categorical_col:
        data[col] = data[col].replace(na_mapping)


not_done_cleaning(train)


def not_tested_cleaning(data: pd.DataFrame):
    """
    To automatically replace all "Not tested" string values within categorical columns with 'NA'.

    Returns
    -------
    None
    """
    na_mapping_nt = {
        'Not tested': 'NA'
    }
    categorical_col_nt = data.select_dtypes(include=['object', 'category']).columns
    for col_nt in categorical_col_nt:
            data[col_nt] = data[col_nt].replace(na_mapping_nt)


not_tested_cleaning(train)


def tbd_cleaning(data: pd.DataFrame):
    """
    To automatically replace all "TBD" string values within categorical columns with 'NA'.

    Returns
    -------
    None
    """
    na_mapping_tbd = {
        'TBD': 'NA'
    }
    categorical_col_tbd = data.select_dtypes(include=['object', 'category']).columns
    for col_tbd in categorical_col_tbd:
            data[col_tbd] = data[col_tbd].replace(na_mapping_tbd)

tbd_cleaning(train)


# EDA and dataset for Catboost

df = pd.DataFrame(train)
# remove column "ID" and "efs_time"
boost_train = train.drop(columns=['ID'])

# Select categorical columns and Find the maximum cardinality
categorical_columns = boost_train.select_dtypes(include=['object', 'category']).columns
cardinality = {col: boost_train[col].nunique() for col in categorical_columns}
max_cardinality = max(cardinality.values()) if cardinality else 0
categorical_list = categorical_columns.tolist()

# Debugging

for col in categorical_list:
    print(f"Unique values in {col}: {boost_train[col].unique()}")

# Print the results
print("Cardinality of categorical features:")
print(cardinality)
print(f"\nMaximum categorical feature cardinality: {max_cardinality}")


for col in categorical_list:
   boost_train[col] = boost_train[col].astype('category')

# print the dtypes attribute
boost_train.dtypes

# Part 2: Classification of efs (ML) Use Catboost to classify efs (1 or 0)

# Prepare the data
train_set, test_set = train_test_split(boost_train, test_size=0.25, random_state=42)
X_train = train_set.drop(columns=['efs_time', 'efs'])
y_train_time = train_set['efs_time']
y_train_event = train_set['efs']
X_test = test_set.drop(columns=['efs_time', 'efs'])
y_test_time = test_set['efs_time']
y_test_event = test_set['efs']


# Create proper label format for AFT
def prepare_aft_labels(time, event):
    """
    For event=1 (uncensored): y_lower = -INF, y_upper = efs_time
    For event=0 (censored): y_lower = efs_time, y_upper = INF
    """
    INF = np.inf
    y_lower_bound = np.where(event == 1,
                             -INF,
                             time.values)

    y_upper_bound = np.where(event == 1,
                             time.values,
                             INF)

    return np.vstack((y_lower_bound, y_upper_bound)).T


# Print some statistics about the survival times
print("\nSurvival Time Statistics:")
print(f"Training set - Min time: {y_train_time.min():.3f}, Max time: {y_train_time.max():.3f}")
print(f"Test set - Min time: {y_test_time.min():.3f}, Max time: {y_test_time.max():.3f}")
print(f"Number of events (Training): {y_train_event.sum()} ({(y_train_event.mean() * 100):.1f}%)")
print(f"Number of events (Test): {y_test_event.sum()} ({(y_test_event.mean() * 100):.1f}%)")

# Prepare training and test labels
y_train_aft = prepare_aft_labels(y_train_time, y_train_event)
y_test_aft = prepare_aft_labels(y_test_time, y_test_event)

# Validate the prepared labels
print("\nPrepared Labels Statistics:")
print(f"Training labels - Min lower bound: {y_train_aft[:, 0].min():.3f}")
print(f"Training labels - Max upper bound: {y_train_aft[:, 1].max():.3f}")
print(f"Test labels - Min lower bound: {y_test_aft[:, 0].min():.3f}")
print(f"Test labels - Max upper bound: {y_test_aft[:, 1].max():.3f}")

# Create DMatrix with the proper label format and enable categorical features
dtrain = xgb.DMatrix(X_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, enable_categorical=True)

# Set the labels correctly
dtrain.set_float_info('label_lower_bound', y_train_aft[:, 0])
dtrain.set_float_info('label_upper_bound', y_train_aft[:, 1])
dtest.set_float_info('label_lower_bound', y_test_aft[:, 0])
dtest.set_float_info('label_upper_bound', y_test_aft[:, 1])

# Fine-tuned parameter ranges based on previous best results
param_grid = {
    'max_depth': (4, 6),  # Narrow around best value 5
    'learning_rate': (0.01, 0.1),  # Lower range since 0.045 worked well
    'min_child_weight': (0.5, 2.0),  # Center around 1.27
    'subsample': (0.5, 0.7),  # Focus near 0.57
    'colsample_bytree': (0.8, 1.0),  # Higher range since 0.88 performed well
    'aft_loss_distribution': ['logistic'],  # Keep best distribution
    'aft_loss_distribution_scale': (7.0, 9.0),  # Center around 8.41
    'num_boost_round': (700, 900)  # Center around 823
}


def xgb_cv_score(params, dtrain, num_boost_round, nfold=5):
    """Run cross-validation for XGBoost AFT model"""
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        metrics={'aft-nloglik'},
        early_stopping_rounds=50,
        seed=42
    )
    return cv_results['test-aft-nloglik-mean'].min()


# Random search
best_score = float('inf')
best_params = None
n_iter = 100

print("Starting random search for AFT model optimization...")
for i in range(n_iter):
    current_params = {
        'objective': 'survival:aft',
        'eval_metric': 'aft-nloglik',
        'tree_method': 'hist',
        'max_depth': random.randint(param_grid['max_depth'][0], param_grid['max_depth'][1]),
        'learning_rate': random.uniform(param_grid['learning_rate'][0], param_grid['learning_rate'][1]),
        'min_child_weight': random.uniform(param_grid['min_child_weight'][0], param_grid['min_child_weight'][1]),
        'subsample': random.uniform(param_grid['subsample'][0], param_grid['subsample'][1]),
        'colsample_bytree': random.uniform(param_grid['colsample_bytree'][0], param_grid['colsample_bytree'][1]),
        'aft_loss_distribution': random.choice(param_grid['aft_loss_distribution']),
        'aft_loss_distribution_scale': random.uniform(param_grid['aft_loss_distribution_scale'][0],
                                                      param_grid['aft_loss_distribution_scale'][1])
    }

    num_boost_round = random.randint(param_grid['num_boost_round'][0], param_grid['num_boost_round'][1])

    try:
        cv_score = xgb_cv_score(current_params, dtrain, num_boost_round)

        if cv_score < best_score:
            best_score = cv_score
            best_params = current_params.copy()
            best_params['num_boost_round'] = num_boost_round

        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{n_iter}, Current best score: {best_score:.4f}")

    except Exception as e:
        print(f"Error in iteration {i + 1}: {str(e)}")
        continue

if best_params is None:
    print("No successful iterations. Please check the data and parameters.")
else:
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=best_params['num_boost_round'],
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # Make predictions
    print("\nMaking predictions...")
    predictions = final_model.predict(dtest)

    # Print model performance
    print("\nFinal Model Details:")
    print("Best Parameters:", best_params)
    print("Best CV Score (aft-nloglik):", best_score)

    # Feature importance analysis
    importance_scores = final_model.get_score(importance_type='gain')
    importance_df = pd.DataFrame([importance_scores]).T
    importance_df.columns = ['importance']
    importance_df = importance_df.sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

    # Save the model
    final_model.save_model('xgboost_aft_model_finetuned.json')



# After your imports, add:

# After creating dtrain and dtest, add:
accuracy_history = []


class PlotIntermediateModel(xgb.callback.TrainingCallback):
    def __init__(self) -> None:
        super().__init__()

    def after_iteration(self, model, epoch, evals_log) -> bool:
        y_pred = model.predict(dtest)
        lower_bound = dtest.get_float_info('label_lower_bound')
        upper_bound = dtest.get_float_info('label_upper_bound')
        acc = np.sum(np.logical_and(y_pred >= lower_bound, y_pred <= upper_bound)) / len(y_pred)
        accuracy_history.append(acc)
        return False

# Modify your final model training:
res = {}
final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=best_params['num_boost_round'],
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    evals_result=res,
    callbacks=[PlotIntermediateModel()],
    verbose_eval=100
)

# Add after model training, before feature importance:
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(res['train']['aft-nloglik'], 'b-o', label='Train')
plt.plot(res['test']['aft-nloglik'], 'r-o', label='Test')
plt.xlabel('Boosting Iterations')
plt.ylabel('Negative Log-Likelihood')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracy_history, 'g-o', label='Accuracy (%)')
plt.xlabel('Boosting Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
