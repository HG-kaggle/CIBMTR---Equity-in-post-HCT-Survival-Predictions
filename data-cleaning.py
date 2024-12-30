import pandas as pd
import numpy as np
import matplotlib as mp

train = pd.read_csv('train.csv')
print(train)

categorical_columns = train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = train.select_dtypes(include=['number']).columns.tolist()

print("Categorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)

# Linear interpolation for numerical columns
train[numerical_columns] = train[numerical_columns].interpolate(method='linear', limit_direction='both')

# Fill categorical columns with most frequent value (mode) in each column
for col in categorical_columns:
    train[col] = train[col].fillna(train[col].mode()[0])

print(train.isnull().sum())

print(train)
# the train file now has no NA
