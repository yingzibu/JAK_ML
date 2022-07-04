from sklearn.neighbors import KNeighborsClassifier
from ast import increment_lineno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def KNN_batch(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    error = []
    for i in range(1, 40):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error.append(np.mean(y_pred != y_test))
        print('for KNN, when neighbor = ', i)
        # print(confusion_matrix(y_test,y_pred))
        # print(classification_report(y_test,y_pred))
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()