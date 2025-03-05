# All the group members have been added to the project
import seaborn as sns
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mp
import xgboost as xgb
import scipy as sp
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score


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
boost_train = boost_train.drop(columns=['efs_time'])

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

# train test split of train
train_set, test_set = train_test_split(boost_train, test_size=0.25, random_state=42)
X_train = train_set.drop(columns=['efs'])
y_train = train_set['efs']
X_test = test_set.drop(columns=['efs'])
y_test = test_set['efs']

# Create DMatrix objects
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

# Define parameter ranges for random search
param_grid = {
    "booster": ["gbtree"],
    'min_child_weight': [random.uniform(1, 3) for _ in range(50)],
    'max_depth': [3, 4, 5],  # Increased depth range
    'eta': [random.uniform(0.06, 0.19) for _ in range(50)],  # Lower learning rate
    'gamma': [random.uniform(0.16, 0.37) for _ in range(50)],
    'max_delta_step': [random.uniform(0.4, 0.7) for _ in range(50)],
    'subsample': [random.uniform(0.63, 0.74) for _ in range(50)],  # Higher subsample
    "colsample_bytree": [random.uniform(0.58, 0.7) for _ in range(50)],  # Higher colsample
    "colsample_bylevel": [random.uniform(0.75, 0.88) for _ in range(50)],
    "colsample_bynode": [random.uniform(0.66, 0.78) for _ in range(50)],
    'num_boost_round': [random.randint(210, 350) for _ in range(100)],  # Fewer rounds
    'alpha': [random.uniform(1.5, 3.0) for _ in range(50)],  # Lower regularization
    'lambda': [random.uniform(2, 3.7) for _ in range(50)]
}


# Function to perform k-fold cross validation
def xgb_cv_score(params, dtrain, num_boost_round=100, nfold=10):
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        metrics='error',
        early_stopping_rounds=50,
        seed=42
    )
    return cv_results['test-error-mean'].min()


# Random search
best_score = float('inf')
best_params = None
n_iter = 100

for i in range(n_iter):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'booster': random.choice(param_grid['booster']),
        'min_child_weight': random.choice(param_grid['min_child_weight']),
        'max_depth': random.choice(param_grid['max_depth']),
        'eta': random.choice(param_grid['eta']),
        'gamma': random.choice(param_grid['gamma']),
        'max_delta_step': random.choice(param_grid['max_delta_step']),
        'subsample': random.choice(param_grid['subsample']),
        'colsample_bytree': random.choice(param_grid['colsample_bytree']),
        'colsample_bylevel': random.choice(param_grid['colsample_bylevel']),
        'colsample_bynode': random.choice(param_grid['colsample_bynode']),
        'alpha': random.choice(param_grid['alpha']),
        'lambda': random.choice(param_grid['lambda'])
    }

    num_boost_round = random.choice(param_grid['num_boost_round'])

    cv_score = xgb_cv_score(params, dtrain, num_boost_round=num_boost_round)

    if cv_score < best_score:
        best_score = cv_score
        best_params = params.copy()
        best_params['num_boost_round'] = num_boost_round

# Train final model with best parameters
final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=best_params['num_boost_round']
)

# Make predictions
y_pred = final_model.predict(dtest)
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
test_accuracy = accuracy_score(y_test, y_pred_binary)

print("Best Parameters:", best_params)
print("Best CV Error:", best_score)
print("Test Accuracy:", test_accuracy)