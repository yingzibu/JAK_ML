import xgboost as xgb
from sklearn.metrics import accuracy_score
from ast import increment_lineno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import shap
def xgboost_batch(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.plots.bar(shap_values, max_display=10)
    shap.plots.heatmap(shap_values)
    shap.summary_plot(shap_values, X_test)