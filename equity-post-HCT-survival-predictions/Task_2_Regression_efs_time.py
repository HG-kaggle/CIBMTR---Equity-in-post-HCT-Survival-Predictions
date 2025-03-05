import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


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
test['hepatic_combined'] = test.apply(combine_hepatic, axis=1)


train = train.drop(columns=['hepatic_severe', 'hepatic_mild'])
test = test.drop(columns=['hepatic_severe', 'hepatic_mild'])


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
test['pulm_combined'] = test.apply(combine_pulm, axis=1)
train = train.drop(columns=['pulm_severe', 'pulm_moderate'])
test = test.drop(columns=['pulm_severe', 'pulm_moderate'])


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
test['type_combined'] = test.apply(combine_type, axis=1)

train = train.drop(columns=['prod_type', 'graft_type'])
test = test.drop(columns=['prod_type', 'graft_type'])

# Drop ID column if it exists
train_efs = train["efs"]
train_efs_time = train["efs_time"]
train = train.drop(columns=['ID', "efs", "efs_time"], errors='ignore')
test_ids = test["ID"]
test = test.drop(columns=['ID'], errors='ignore')

categorical_columns = train.select_dtypes(exclude=[np.number]).columns
numerical_columns = train.select_dtypes(include=[np.number]).columns
test_categorical = test.select_dtypes(exclude=[np.number]).columns
test_numerical = test.select_dtypes(include=[np.number]).columns

# Handling missing values
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Fit and transform only on the feature columns (excluding 'efs' and 'efs_time')
train[numerical_columns] = num_imputer.fit_transform(train[numerical_columns])
test[test_numerical] = num_imputer.transform(test[test_numerical])
train[categorical_columns] = cat_imputer.fit_transform(train[categorical_columns])
test[test_categorical] = cat_imputer.transform(test[test_categorical])

# Encoding categorical variables using Target Encoding
encoder = TargetEncoder(cols=categorical_columns)
train[categorical_columns] = encoder.fit_transform(train[categorical_columns], train_efs)
test[categorical_columns] = encoder.transform(test[categorical_columns])
train["efs"] = train_efs
train["efs_time"] = train_efs_time

# Split train data into train and validation
X = train.drop(columns=['efs', 'efs_time'])
y_time = train_efs_time
y_event = train_efs
X_train, X_valid, y_train_time, y_valid_time, y_train_event, y_valid_event = train_test_split(
    X, y_time, y_event, test_size=0.25, random_state=42
)

# Prepare AFT labels
def prepare_aft_labels(time, event):
    INF = np.inf
    y_lower = np.where(event == 1, -INF, time.values)
    y_upper = np.where(event == 1, time.values, INF)
    return np.vstack((y_lower, y_upper)).T


y_train_aft = prepare_aft_labels(y_train_time, y_train_event)
y_valid_aft = prepare_aft_labels(y_valid_time, y_valid_event)

# Create DMatrix including 'efs' and 'efs_time'
dtrain = xgb.DMatrix(X_train, label=y_train_time, enable_categorical=True)
dvalid = xgb.DMatrix(X_valid, label=y_valid_time, enable_categorical=True)
dtest = xgb.DMatrix(test, enable_categorical=True)

dtrain.set_float_info('label_lower_bound', y_train_aft[:, 0])
dtrain.set_float_info('label_upper_bound', y_train_aft[:, 1])
dvalid.set_float_info('label_lower_bound', y_valid_aft[:, 0])
dvalid.set_float_info('label_upper_bound', y_valid_aft[:, 1])

# Set parameters with wider tuning range
param_grid = {
    'max_depth': (2, 3),
    'learning_rate': (0.01, 0.1),
    'min_child_weight': (0.3, 0.65),
    'subsample': (0.6, 0.75),
    'colsample_bytree': (0.97, 1),
    'aft_loss_distribution': ['logistic'],
    'aft_loss_distribution_scale': (9.9, 10.0),
    'alpha': (0, 0.08),
    'lambda': (0, 0.1),
    'num_boost_round': (900, 1000)
}

def xgb_cv_score(params, dtrain, num_boost_round, nfold=5):
    """Run cross-validation for XGBoost AFT model"""
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        metrics={'aft-nloglik'},
        early_stopping_rounds=100,
        seed=42
    )
    return cv_results['test-aft-nloglik-mean'].min()

# Random search optimization
best_score = float('inf')
best_params = None
n_iter = 300

print("Starting random search for AFT model optimization...")
for i in range(n_iter):
    current_params = {
        'objective': 'survival:aft',
        'eval_metric': 'aft-nloglik',
        'tree_method': 'hist',
        'max_depth': random.randint(*param_grid['max_depth']),
        'learning_rate': random.uniform(*param_grid['learning_rate']),
        'min_child_weight': random.uniform(*param_grid['min_child_weight']),
        'subsample': random.uniform(*param_grid['subsample']),
        'colsample_bytree': random.uniform(*param_grid['colsample_bytree']),
        'aft_loss_distribution': random.choice(param_grid['aft_loss_distribution']),
        'aft_loss_distribution_scale': random.uniform(*param_grid['aft_loss_distribution_scale']),
        'alpha': random.uniform(*param_grid['alpha']),
        'lambda': random.uniform(*param_grid['lambda'])
    }

    num_boost_round = random.randint(*param_grid['num_boost_round'])

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

if best_params:
    print(best_params)
    print("\nTraining final model with best parameters...")
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=best_params['num_boost_round'],
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=200
    )
    final_model.save_model('xgboost_aft_model_finetuned.json')
    efs_time = final_model.predict(dtest)
    test["prediction"] = efs_time
    original_test = pd.read_csv('test.csv')

    # Add predictions to the test DataFrame
    original_test['efs_time'] = efs_time  # Store predictions

    # Save the updated test dataset
    original_test.to_csv("test_with_efs_time.csv", index=False)







