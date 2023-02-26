from sklearn.ensemble import RandomForestClassifier
from function import evaluate
from tkinter import Grid
from function import evaluate, load_data, get_tpr_fpr, save_tpr_fpr, load_tpr_fpr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
from function import evaluate,get_preds,save_tpr_fpr
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
from sklearn import metrics

def RF_single(X, y, n_estimator, enzyme, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=42)
    RF = RandomForestClassifier(n_estimators=n_estimator)
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    y_prob = RF.predict_proba(X_test)
    evaluate(y_test, y_pred, y_prob)
    tpr_values, fpr_values = get_tpr_fpr(y_test, y_prob)
    save_tpr_fpr('RF', enzyme, tpr_values, fpr_values)
    # metrics.plot_roc_curve(RF, X_test, y_test) 
    
    filename = 'RF_' + enzyme +'.sav'
    model_path = 'model/'
    modelname = model_path + filename
    pickle.dump(RF, open(modelname, 'wb'))
