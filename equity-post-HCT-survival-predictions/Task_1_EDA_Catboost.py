# All the group members have been added to the project
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mp
import xgboost as xgb
import scipy as sp
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


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

# Combine strongly related columns

# Combine 'hepatic_severe' & 'hepatic_mild'
def combine_hepatic(row):
    if row['hepatic_severe'] == 'Yes':
        return 'Severe'
    elif row['hepatic_mild'] == 'Yes':
        return 'Mild'
    elif row['hepatic_severe'] == 'NA' and row['hepatic_mild'] == 'NA':
        return 'NA'
    else:
        return 'Light'

train['hepatic_combined'] = train.apply(combine_hepatic, axis=1)

train = train.drop(columns=['hepatic_severe', 'hepatic_mild'])

# Combine 'pulm_severe' & 'pulm_mild'
def combine_pulm(row):
    if row['pulm_severe'] == 'Yes':
        return 'Severe'
    elif row['pulm_mild'] == 'Yes':
        return 'Mild'
    elif row['pulm_severe'] == 'NA' and row['pulm_mild'] == 'NA':
        return 'NA'
    else:
        return 'Light'

train['pulm_combined'] = train.apply(combine_pulm, axis=1)

train = train.drop(columns=['pulm_severe', 'pulm_mild'])

# Combine prod_type & graft_type
def combine_type(row):
    if row['prod_type'] == 'BM' and row['graft_type'] == 'Bone marrow':
        return 'BM'
    elif row['prod_type'] == 'PB' and row['graft_type'] == 'Peripheral blood':
        return 'PB'
    elif row['prod_type'] == 'BM' and row['graft_type'] == 'Peripheral blood':
        return 'BM/PB'
    elif row['prod_type'] == 'PB' and row['graft_type'] == 'Bone marrow':
        return 'PB/BM'
    else:
        return 'NA'

train['type_combined'] = train.apply(combine_type, axis=1)

train = train.drop(columns=['prod_type', 'graft_type'])

# EDA and dataset for Catboost

df = pd.DataFrame(train)
# remove column "ID" and "efs_time"
cat_train = train.drop(columns=['ID'])
cat_train = cat_train.drop(columns=['efs_time'])

# Select categorical columns and Find the maximum cardinality
categorical_columns = cat_train.select_dtypes(include=['object', 'category']).columns
cardinality = {col: cat_train[col].nunique() for col in categorical_columns}
max_cardinality = max(cardinality.values()) if cardinality else 0
categorical_list = categorical_columns.tolist()

# Debugging

for col in categorical_list:
    print(f"Unique values in {col}: {cat_train[col].unique()}")

# Print the results
print("Cardinality of categorical features:")
print(cardinality)
print(f"\nMaximum categorical feature cardinality: {max_cardinality}")

# Delete the rows with less than 80% Completeness (by NA & -1)
# cat_train = cat_train[((train.eq('NA') | train.eq(-1)).sum(axis = 1) < 23)]

# Part 2: Classification of efs (ML) Use Catboost to classify efs (1 or 0)

# train test split of train with efs = 1
train_set, test_set = train_test_split(cat_train, test_size=0.25, random_state=42)
X_train = train_set.drop(columns=['efs'])  # Features of train data
y_train = train_set['efs']  # Outcome of train data
X_test = test_set.drop(columns=['efs'])  # Features of test data
y_test = test_set['efs']  # Target of test data
# Convert the dataframes to numpy arrays (CatBoost works well with Pool format)
train_pool = Pool(X_train, label=y_train, cat_features=categorical_list)
test_pool = Pool(X_test, label=y_test, cat_features=categorical_list)

# Use CatBoost

# Define the parameter grid for the random grid search
param_grid = {
    'iterations': randint(1300, 1400),
    'depth': [3],
    'learning_rate': uniform(0.001, 0.06),
    'l2_leaf_reg': uniform(5, 6),
    'border_count': [240],  # Number of splits for numerical features
}

model = CatBoostClassifier(
    loss_function='Logloss',
    verbose=True, # Suppress training output for readability
    one_hot_max_size=20,
    cat_features=categorical_list,
    #task_type="GPU" # Do not enable GPU unless you are on a GPU server
)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=100,  # Number of random samples of parameter grid to try, we can try 50 or even 100.
    scoring='accuracy',  # Metric for evaluation
    cv=10,  # Number of cross-validation folds
    verbose=1,  # Show progress
    random_state=42,  # Reproducibility
    n_jobs=-1)  # Use all available CPU processors

random_search.fit(X_train, y_train)
print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)

# Initialize the CatBoost model with the best parameters
best_params = random_search.best_params_
model = CatBoostClassifier(**best_params, loss_function='Logloss', verbose=False)

# Train the model with the training pool and evaluate on the test pool
model.fit(train_pool, eval_set=test_pool, verbose=False)

# Optional: If you want to access final predictions
y_pred = model.predict(test_pool)
