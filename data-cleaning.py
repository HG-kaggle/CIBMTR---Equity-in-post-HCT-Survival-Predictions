# Author Yang Xiang, Date 2024 12 30.
# All the group members have been added to the project
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mp
import xgboost as xgb
import scipy as sp
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform
from gower import gower_matrix

# Part 1: Data cleaning, adding -1 for numerical missing values, and "NA" string value
# for categorical missing values.

train = pd.read_csv('equity-post-HCT-survival-predictions/train.csv')
# This is for test purpose to fix

## For de-bugging purpose, we first select 10 rows of data to try running agglo algorithm.
# train = train.head(10)

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
    categorical_col = data.select_dtypes(exclude=[np.number]).columns
    for col_ID in categorical_col:
        if 'Not done' in data[col_ID]:
            data[col_ID] = data[col_ID].replace(na_mapping)


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
    categorical_col_nt = data.select_dtypes(exclude=[np.number]).columns
    for col_nt in categorical_col_nt:
        if 'Not tested' in data[col_nt]:
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
    categorical_col_tbd = data.select_dtypes(exclude=[np.number]).columns
    for col_tbd in categorical_col_tbd:
        if 'TBD' in data[col_tbd]:
            data[col_tbd] = data[col_tbd].replace(na_mapping_tbd)


tbd_cleaning(train)

# Part 2: Scale the data and apply Agglomerative Clustering.

# 1 scaling data
scaler = StandardScaler()
train_numeric_scaled = pd.DataFrame(scaler.fit_transform(train[numerical_columns]),
                                    columns=numerical_columns, index=train.index)
train_categorical = train[categorical_columns]
train_scaled = pd.concat([train_numeric_scaled, train_categorical], axis=1)
train_scaled = train_scaled[train.columns]

# 2 Fitting agglo
# Use Hamming distance (for multi-class nominal categorical data, but hamming only works for
# discrete numerical values.)

# Calculate Gower distance matrix
distance_matrix = gower_matrix(train_scaled)


# Apply Agglomerate Clustering
agglo = AgglomerativeClustering(n_clusters=4, metric='precomputed', linkage='average')
clusters = agglo.fit_predict(distance_matrix)

# Add cluster labels to the original dataset
train_scaled['Cluster'] = clusters

# Print the first few rows of the dataset with cluster labels
print(train_scaled.head())

# Extract cluster




