import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os

sys.path.append(os.path.abspath(".."))

train_data = pd.read_csv('../data/train.csv', encoding='windows-1251', sep=',', low_memory=False)
train_data = train_data.iloc[:-16184]

from feature.feature_engeeniring import filter_family_size

train_data = filter_family_size(train_data, 10)

from feature.feature_engeeniring import impute_with_regression

train_data = impute_with_regression(train_data, 'MonthlyIncome')

from feature.feature_engeeniring import delete_outliers

continuous_cols = ['DebtRatio', 'MonthlyIncome', 'TotalBalanceDivideCreditLimits']
train_data = delete_outliers(train_data, continuous_cols, 0.99)

from feature.feature_engeeniring import generate_features

train_data = generate_features(train_data)

from sklearn.model_selection import train_test_split

drop_cols = [
    'PastDueLess_60',
    'PastDue60_90',
    'PastDue90_More',
    'OpenLoans',
    'RealEstateLoans',
    'Income_inverse',
    'HasPastDue',
    'HasRealEstateLoan',
    'Age_inverse'
]

X = train_data.drop(['Target'] + drop_cols, axis=1)
y = train_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Определим функцию для оптимизации гиперпараметров с использованием Optuna
def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'device': 'gpu',  # включаем GPU
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 10, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 10, log=True),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
    }



    model = lgb.LGBMClassifier(**param)

    # Обучаем модель
    model.fit(X_train, y_train)

    # Предсказания на тестовых данных
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Метрики
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Оптимизируем модель по ROC AUC
    return roc_auc

# Запуск оптимизации
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=250)

print("Best hyperparameters:", study.best_params)

# Лучшие параметры
best_params = study.best_params
# Строим модель с лучшими гиперпараметрами
best_model = lgb.LGBMClassifier(**best_params)
# Обучаем финальную модель на всех данных
best_model.fit(X, y)

# Оценка модели на тестовых данных
y_pred_final = best_model.predict(X_test)
y_prob_final = best_model.predict_proba(X_test)[:, 1]
final_f1 = f1_score(y_test, y_pred_final)
final_roc_auc = roc_auc_score(y_test, y_prob_final)

print("Final F1 Score:", final_f1)
print("Final ROC AUC:", final_roc_auc)

