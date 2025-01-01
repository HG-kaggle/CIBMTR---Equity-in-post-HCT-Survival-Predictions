# Author Yang Xiang, Date 2024 12 30.

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mp
import xgboost as xgb
import scipy as sp
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import RandomizedSearchCV

train = pd.read_csv('equity-post-HCT-survival-predictions/train.csv')
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

# filter out imputed train with efs == 1.0
df = pd.DataFrame(train)
efs1_train = df[df['efs'] == 1.0]

# remove column "ID"
efs1_train = efs1_train.drop(columns=['ID'])

object_columns = efs1_train.select_dtypes(include=['object']).columns

for col in object_columns:
    efs1_train[col] = pd.to_numeric(efs1_train[col], errors='coerce')

# train test split of train with efs = 1

train_set, test_set = train_test_split(efs1_train, test_size=0.25, random_state=42)

X_train = train_set.drop(columns=['efs_time'])  # Features of train data
y_train = train_set['efs_time']  # Outcome of train data

X_test = test_set.drop(columns=['efs_time'])  # Features of test data
y_test = test_set['efs_time']  # Target of test data

# Convert the dataframes to numpy arrays (CatBoost works well with Pool format)
train_pool = Pool(X_train, label=y_train)
test_pool = Pool(X_test, label=y_test)



# Use CatBoost

# Define the parameter grid for the random grid search
param_grid = {
    'iterations': [100, 500, 1000, 1500],
    'depth': [5, 8, 10, 15],
    'learning_rate': [0.001, 0.01, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128],  # Number of splits for numerical features
}

model = CatBoostClassifier(
    loss_function='Logloss',
    verbose=False  # Suppress training output for readability
)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=20,  # Number of random samples to try
    scoring='accuracy',  # Metric for evaluation
    cv=10,  # Number of cross-validation folds
    verbose=1,  # Show progress
    random_state=42,  # Reproducibility
    n_jobs=-1  # Use all available processors
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)


