# Author Yang Xiang, Date 2024 12 30.

import pandas as pd
import numpy as np
import matplotlib as mp
import xgboost as xgb
import scipy as sp

train = pd.read_csv('train.csv')
# This is for test purpose to fix 

categorical_columns = train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = train.select_dtypes(include=['number']).columns.tolist()

print("Categorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)

# Linear interpolation for numerical columns in data
train[numerical_columns] = train[numerical_columns].interpolate(method='linear', limit_direction='both')

# Fill categorical columns with most frequent value (mode) in each column
# (we assume that the missing columns should be imputed with mode value, we would check back later
# whether this is appropriate）
for col in categorical_columns:
    train[col] = train[col].fillna(train[col].mode()[0])

print(train.isnull().sum())

