from function import *
from jak_dataset import *
from sklearn.neighbors import KNeighborsClassifier
from ast import increment_lineno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
import pickle
import os

header = ['bit'+str(i) for i in range(167)]
# import os
# def create_path(path):
#     isExist = os.path.exists(path)
#     print(path, ' folder is in directory: ', isExist)
#     if not isExist:
#         os.makedirs(path)
#         print(path, " is created!")

model_path = "model_0.25nM/"
create_path(model_path)


def KNN_rough_tune(data, enzyme):
    # global header
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
    path = "knn_figures/"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    #printing if the path exists or not
    print(path, ' folder is in directory: ', isExist)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(path)
        print(path, " is created!")
    plt.savefig(path + enzyme + '_rough_tune.png')
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

def KNN_fine_tune(data, enzyme='JAK1', rough_neighbor=10):
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

    max_accuracy = max(k_scores)
    neighbors_rough = []
    for i, j in zip(k_list, k_scores):
        if j == max_accuracy:
            print('Fine tune max_accuracy occurs at neighbor #', i)   
            neighbors_rough.append(i)
    plt.plot(k_list, k_scores)
    plt.xlabel('Value of neighbor k for KNN')
    plt.ylabel('5-fold cross-validated accuracy')
    plt.title(f'{enzyme} KNN accuracy with neighbors fine tune, best_neighbor = {neighbors_rough[0]}')

    path = "knn_figures/"
    isExist = os.path.exists(path) #printing if the path exists or not
    print(path, ' folder is in directory: ', isExist)
    if not isExist: # Create a new directory because it does not exist
        os.makedirs(path)
        print(path, " is created!")

    plt.savefig(path + enzyme + '_fine_tune.png')

    print('neighbors_fine for ', enzyme, ' : ', neighbors_rough)
    print('max accuracy: ', max_accuracy)
    return neighbors_rough[0]

data_path = 'Data_MTATFP/data_label_0.25/'
for enzyme in ['JAK1', 'JAK2', 'JAK3', 'TYK2']:
    
    data_partial = pd.read_csv(data_path + enzyme + '_partial.csv')
    data_test = pd.read_csv(data_path + enzyme + '_test.csv')
    
    data = JAK_dataset(data_partial['SMILES'], data_partial['Activity'])
    data_test_jak = JAK_dataset(data_test['SMILES'], data_test['Activity'])

    top_k = KNN_rough_tune(data.get_df(), enzyme)
    final_top_k = KNN_fine_tune(data.get_df(), enzyme, top_k)
    print(enzyme, ', best neighbor is ', final_top_k)
    
    knn = KNeighborsClassifier(n_neighbors=final_top_k)
    knn.fit(data.get_MACCS(), data.get_activity())
    knn.probability=True
    y_pred = knn.predict(data_test_jak.get_MACCS())
    y_prob = knn.predict_proba(data_test_jak.get_MACCS())
    print('for ', enzyme)
    evaluate(data_test_jak.get_activity(), y_pred, y_prob)
    filename = model_path + 'knn_' + enzyme + '.pkl'
    pickle.dump(knn, open(filename, 'wb'))