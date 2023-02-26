from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
from function import evaluate,get_preds,save_tpr_fpr
import pickle

global header
header = ['bit'+str(i) for i in range(167)]

def SVM_batch(enzyme):
    path = 'data/' + enzyme + '_' + 'MACCS.csv'
    data = pd.read_csv(path)
    header = ['bit'+str(i) for i in range(167)]
    X = data[header]
    y = data['Activity']
    for k in ['linear', 'poly', 'rbf', 'sigmoid']:
        if k =='poly':
            model = SVC(kernel=k, degree=8)
        else:
            model = SVC(kernel=k)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
        random_state=42, test_size = 0.2)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print('SVM: ',k, 'Scores: ', scores.mean())
        model.probability=True
        model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, 
        random_state=42, test_size = 0.5)
        y_prob = model.predict_proba(X_test)
        proba = y_prob[:, 1]
        y_pred = get_preds(0.5, proba)
        evaluate(y_test, y_pred, y_prob)

        # Get AUC values for AUC figure
        roc_values = []
        for thresh in np.linspace(0, 1, 100):
            preds = get_preds(thresh, proba)
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            tpr = tp/(tp+fn)
            fpr = fp/(fp+tn)
            roc_values.append([tpr, fpr])
        tpr_values, fpr_values = zip(*roc_values)
        
        save_tpr_fpr('SVM_'+k, enzyme, tpr_values, fpr_values)

        filename = 'model/SVM_'+k+'_'+enzyme+'.sav'
        pickle.dump(model, open(filename, 'wb'))

enzymes = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
for enzyme in enzymes: 
    print('################################################')
    print('FOR', enzyme)
    SVM_batch(enzyme)
