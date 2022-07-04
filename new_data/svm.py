from ast import increment_lineno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#%matplotlib inline
# Input: X, y as label
# Output: SVM prediction with different kernels 
def SVM_batch(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    for k in ['linear', 'poly', 'rbf', 'sigmoid']:
        if k =='poly':
            model = SVC(kernel=k, degree=8)
        else:
            model = SVC(kernel=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print('For SVM model with ', k)
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))


