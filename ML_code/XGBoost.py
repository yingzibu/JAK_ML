from tkinter import Grid
import xgboost as xgb
from function import evaluate, load_data, save_tpr_fpr, load_tpr_fpr, get_tpr_fpr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
from function import evaluate,get_preds,save_tpr_fpr
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle

global header
header = ['bit'+str(i) for i in range(167)]
path = ''
model_path = path + 'model/'
# enzyme = 'JAK1'
for enzyme in ['JAK1', 'JAK2', 'JAK3', 'TYK2']:
    X, y = load_data(enzyme)
    # params = {
    #         'max_depth': [3, 6],
    #         'learning_rate': [0.1],
    #         'n_estimators': [10],
    #         'colsample_bylevel': [0, 1],
    #         'colsample_bytree': [0, 1]
    #     }

    params = {
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [10, 50, 100, 500, 1000],
        'colsample_bylevel': [0.3, 0.7, 1],
        'colsample_bytree': [0.3, 0.7, 1]
    }

    model = xgb.XGBClassifier()
    clf = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', verbose=1)
    clf.fit(X, y)
    print('Best estimators:', clf.best_params_)
    file = 'XGBoost_params/XGBoost_' + enzyme + '.pickle'
    with open(file, 'wb') as f:
        pickle.dump(clf.best_params_, f)
    print('Best accuracy:', clf.best_score_)
    with open(file, 'rb') as f:
        tuned_params = pickle.load(f)

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=42)
    model = xgb.XGBClassifier(**tuned_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    evaluate(y_test, y_pred, y_prob)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    filename = 'XGBoost_' + enzyme +'.sav'
    pickle.dump(model, open(model_path + filename, 'wb'))

