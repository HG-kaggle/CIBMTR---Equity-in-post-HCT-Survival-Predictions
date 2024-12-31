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
# train[numerical_columns] = train[numerical_columns].interpolate(method="polynomial", order=2)


# Fill categorical columns with most frequent value (mode) in each column
# (we assume that the missing columns should be imputed with mode value, we would check back later
# whether this is appropriate）
for col in categorical_columns:
    train[col] = train[col].fillna(train[col].mode()[0])

print(train.isnull().sum())

# Part 2 Since besides efs and efs_time, we have remaining 57 variables, which is feasible to analyze the trend
# and perform the classification work. As in the project efs=0 means we do not know whether the observation
# confront the event, this step aims to classify the efs=0 population with "highly probable" efs=1
# population and the others remain with efs=0.

