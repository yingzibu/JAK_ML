from sklearn.neighbors import KNeighborsClassifier
from ast import increment_lineno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
from function import evaluate, load_tpr_fpr, save_tpr_fpr
from function import load_model, test, smile_list_to_MACCS, get_preds
import pickle

global header
header = ['bit'+str(i) for i in range(167)]

def KNN_rough_tune(enzyme = 'JAK1'):
    path = 'data/' + enzyme + '_' + 'MACCS.csv'
    data = pd.read_csv(path)
    header = ['bit'+str(i) for i in range(167)]
    X = data[header]
    y = data['Activity']
    k_range = range(1, 50)
    k_list = [item * 10 for item in k_range]
    print('First rough test neighbors 1-500')
    k_scores = []
    # Cross validation 5
    for k in k_list:
        if k%50 == 0:
            print('Now trying neighbor#', k)
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())

    plt.plot(k_list, k_scores)
    plt.xlabel('Value of neighbor k for KNN')
    plt.ylabel('5-fold cross-validated accuracy')
    title = enzyme + ' KNN accuracy with neighbors ' + 'rough tune'
    plt.title(title)
    plt.savefig('figures/' + enzyme + '_rough_tune.png')
    plt.close()
    print('Rough tune max k_scores: ')
    print(max(k_scores))
    max_accuracy = max(k_scores)
    neighbors_rough = []
    for i, j in zip(k_list, k_scores):
        if j == max_accuracy:
            print('Rough tune max_accuracy occurs at neighbor #', i)   
            neighbors_rough.append(i)
            
            # Further fine tune

    print('neighbors_rough for ', enzyme, ' : ', neighbors_rough)
    print(' max accuracy: ', max_accuracy)
    return neighbors_rough[0]

def KNN_fine_tune(enzyme='JAK1', rough_neighbor=10):
    path = 'data/' + enzyme + '_' + 'MACCS.csv'
    data = pd.read_csv(path)
    header = ['bit'+str(i) for i in range(167)]
    X = data[header]
    y = data['Activity']
    low_bound = max(rough_neighbor - 10, 1)
    up_bound = low_bound + 20 
    k_range = range(low_bound, up_bound)
    k_scores = []
    for k in k_range:
        if k%5 == 0:
            print('Now trying neighbor ', k)
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    k_list = [i for i in k_range]
    
    print('Fine tune max k_scores: ')
    print(max(k_scores))
    max_accuracy = max(k_scores)
    neighbors_rough = []
    for i, j in zip(k_list, k_scores):
        if j == max_accuracy:
            print('Fine tune max_accuracy occurs at neighbor #', i)   
            neighbors_rough.append(i)
    plt.plot(k_list, k_scores)
    plt.xlabel('Value of neighbor k for KNN')
    plt.ylabel('5-fold cross-validated accuracy')
    title = enzyme + ' KNN accuracy with neighbors ' + 'fine tune, best_neighbor = ' + str(neighbors_rough[0])
    plt.title(title)
    plt.savefig('figures/' + enzyme + '_fine_tune.png')

    print('neighbors_fine for ', enzyme, ' : ', neighbors_rough)
    print('max accuracy: ', max_accuracy)
    return neighbors_rough[0]

# best_neighbors = []
# for enzyme in ['JAK1', 'JAK2', 'JAK3', 'TYK2']:
#     neighbor = KNN_rough_tune(enzyme)
#     best_neighbor = KNN_fine_tune(enzyme, neighbor)
#     best_neighbors.append(best_neighbor)

enzymes = ['JAK1', 'JAK2', 'JAK3', 'TYK2']
best_neighbors = [1,1,1,3]

for i in range(4): 
    enzyme = enzymes[i]
    best_neighbor = best_neighbors[i]
    file = 'data/' + enzyme + '_MACCS.csv'
    data = pd.read_csv(file)
    X = data[header]
    y = data['Activity']
    knn = KNeighborsClassifier(n_neighbors=best_neighbor)
    X_train, X_test, y_train, y_test= train_test_split(X, y, 
                                      test_size = 0.2, random_state=42)
    
    knn.fit(X_train, y_train) 
    knn.probability=True  
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)
    print('for ', enzyme)
    evaluate(y_test, y_pred, y_prob)
    filename = 'model/knn_' + enzyme
    pickle.dump(knn, open(filename, 'wb'))

    roc_values = []
    
    for thresh in np.linspace(0,1,10000):
        preds = get_preds(thresh, y_prob[:, 1])
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        tpr = tp /(tp+fn)
        fpr = fp/(fp+tn)
        roc_values.append([tpr, fpr])
    tpr_values, fpr_values = zip(*roc_values)
    save_tpr_fpr('knn', enzyme, tpr_values, fpr_values)










