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


## Part 1: Data cleaning, adding -1 for numerical missing values, and "NA" string value
## for categorical missing values.

train = pd.read_csv('equity-post-HCT-survival-predictions/train.csv')
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




