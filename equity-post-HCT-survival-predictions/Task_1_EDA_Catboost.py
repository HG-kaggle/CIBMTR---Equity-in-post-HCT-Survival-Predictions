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

train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
validation= pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

categorical_columns = train.select_dtypes(exclude=[np.number]).columns
numerical_columns = train.select_dtypes(include=[np.number]).columns
validation_categorical_columns = validation.select_dtypes(exclude=[np.number]).columns
validation_numerical_columns = validation.select_dtypes(include=[np.number]).columns

# Linear interpolation for numerical columns in data: we transform all NA into -1.
for col in numerical_columns:
    if train[col].isna().sum() > 0:
        train[col] = train[col].fillna(-1)

for col in validation_numerical_columns:
    if validation[col].isna().sum() > 0:
        validation[col] = validation[col].fillna(-1)
# Fill categorical columns with string "NA" value.

for col in categorical_columns:
    if train[col].isna().sum() > 0:
        train[col] = train[col].fillna('NA')

for col in validation_categorical_columns:
    if validation[col].isna().sum() > 0:
        validation[col] = validation[col].fillna('NA')

# Check if the processed train dataset has any missing values.
print(train.isnull().sum())
print(validation.isnull().sum())

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
validation["dri_score"] = validation["dri_score"].replace(na_mapping_dri)

# Replace NA strings for cyto_score:
na_mapping_cyto = {
    'TBD': 'NA',
    'Other': 'NA',
    'Not tested': 'NA'
}
train['cyto_score'] = train['cyto_score'].replace(na_mapping_cyto)
validation["cyto_score"] = validation["cyto_score"].replace(na_mapping_cyto)

# Replace NA string for conditioning_intensity:
na_mapping_ci = {
    'N/A, F(pre-TED) not submitted': 'NA',
    'No drugs reported': 'NA'
}
train['conditioning_intensity'] = train['conditioning_intensity'].replace(na_mapping_ci)
validation['conditioning_intensity'] = validation['conditioning_intensity'].replace(na_mapping_ci)

# Impute not used value string for melphalan_dose:
na_mapping_mel = {
    "N/A, Mel not given": 'Not Used'
}
train['melphalan_dose'] = train['melphalan_dose'].replace(na_mapping_mel)
validation['melphalan_dose'] = validation['melphalan_dose'].replace(na_mapping_mel)


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
not_done_cleaning(validation)


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
not_tested_cleaning(validation)


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
tbd_cleaning(validation)
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

# Combine 'pulm_severe' & 'pulm_moderate'
def combine_pulm(row):
    if row['pulm_severe'] == 'Yes':
        return 'Severe'
    elif row['pulm_moderate'] == 'Yes':
        return 'Mild'
    elif row['pulm_severe'] == 'NA' and row['pulm_moderate'] == 'NA':
        return 'NA'
    else:
        return 'Light'

train['pulm_combined'] = train.apply(combine_pulm, axis=1)

train = train.drop(columns=['pulm_severe', 'pulm_moderate'])

# Combine 'prod_type' & 'graft_type'
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
df_validate = pd.DataFrame(validation)
# remove column "ID" and "efs_time"
cat_train = train.drop(columns=['ID'])
cat_validation = validation.drop(columns=['ID'])

# Select categorical columns and Find the maximum cardinality
categorical_columns = cat_train.select_dtypes(include=['object', 'category']).columns
cardinality = {col: cat_train[col].nunique() for col in categorical_columns}
max_cardinality = max(cardinality.values()) if cardinality else 0
categorical_list = categorical_columns.tolist()

for col in categorical_list:
    print(f"Unique values in {col}: {cat_train[col].unique()}")

# Print the results
print("Cardinality of categorical features:")
print(cardinality)
print(f"\nMaximum categorical feature cardinality: {max_cardinality}")

# Part 2: Classification of efs (ML) Use Catboost to classify efs (1 or 0)

# train test split of train with efs = 1
train_set, test_set = train_test_split(cat_train, test_size=0.25, random_state=42)
y_train = train_set['efs']
X_train = train_set.drop(columns=['efs'])  # Features of train data
y_test = test_set['efs']
X_test = test_set.drop(columns=['efs'])  # Features of test data
# Convert the dataframes to numpy arrays (CatBoost works well with Pool format)
train_pool = Pool(X_train, label=y_train, cat_features=categorical_list)
test_pool = Pool(X_test, label=y_test, cat_features=categorical_list)
# Use CatBoost
# Define the parameter grid for the random grid search
param_grid = {
    'iterations': randint(1250, 1400),
    'depth': [3],
    'learning_rate': uniform(0.048, 0.07),
    'l2_leaf_reg': uniform(5.2, 5.5),
    'border_count':[240],
    # CYCLE 10
}

model = CatBoostClassifier(
    loss_function='Logloss',
    verbose=False, # Suppress training output for readability
    one_hot_max_size=20,
    cat_features=categorical_list,
    train_dir=None,
    allow_writing_files=False,
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
validation_pool = Pool(validation, cat_features=categorical_list)
y_pred = model.predict(validation_pool, verbose = False)
# Convert predictions to DataFrame
validation['efs'] = y_pred.astype(int)  # Ensure predictions are integers
# Load original test.csv file
test_original = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# Merge validation data with original test.csv based on 'ID'
merged_test = test_original.merge(validation[['ID', 'efs_pred']], on='ID', how='left')
# Save the merged file
merged_test.to_csv("/kaggle/working/test_with_efs.csv", index=False)
print("Predictions added and saved as test_with_efs.csv!")