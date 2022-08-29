import pandas as pd
import pubchempy as pcp
from os import walk
import os 
from collections import OrderedDict
import numpy as np
import math
import json
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from rdkit.Chem import MACCSkeys
from rdkit import Chem
import torch
import pickle
from sklearn.model_selection import train_test_split

def smile_list_to_MACCS(smi_list):
    MACCS_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        maccs = list(MACCSkeys.GenMACCSKeys(mol).ToBitString())
        MACCS_list.append(maccs)
    return MACCS_list

def get_preds(threshold, probabilities):
    try:
        if probabilities.shape[1] == 2:
            probabilities = probabilities[:, 1]
    except:
        pass
    return [1 if prob > threshold else 0 for prob in probabilities]

def evaluate_model(TP, FP, TN, FN):
    
    ACCURACY = (TP + TN) / (TP+FP+TN+FN)
    SE = TP/(TP+FN)
    recall = SE
    SP = TN/(TN+FP)
    weighted_accuracy = (SE + SP) / 2

    precision = TP / (TP + FP)
    SP = TN/(TN+FP)    
    F1 = 2 * precision * recall /(precision + recall)

    temp = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    if temp != 0:
        MCC = (TP*TN-FP*FN)*1.0/(math.sqrt(temp))
    else:
        print('equation for MCC is (TP*TN-FP*FN)*1.0/(math.sqrt(temp))')
        print('TP, FP, TN, FN', TP, FP, TN, FN)
        print('temp=0')
        MCC = 'N/A'

    return ACCURACY,SE, SP, weighted_accuracy, precision, F1, MCC 
    

def evaluate(y_real, y_hat, y_prob):
    TN, FP, FN, TP = confusion_matrix(y_real, y_hat).ravel()
    ACCURACY,SE, SP, weighted_accuracy, precision, F1, MCC  = evaluate_model(TP, FP, TN, FN)
    try:
        if y_prob.shape[1] == 2:
            proba = y_prob[:, 1]
        else:
            proba = y_prob
    except:
        proba = y_prob
    AP = average_precision_score(y_real, proba)
    AUC = roc_auc_score(y_real, proba)
    print('Accuracy, weighted accuracy, precision, recall/SE, SP,     F1,     AUC,     MCC,     AP')
    if MCC != 'N/A':
        print("& %5.3f" % (ACCURACY), " &%7.3f" % (weighted_accuracy), " &%15.3f" % (precision), 
      " &%10.3f" % (SE), " &%5.3f" % (SP), " &%5.3f" % (F1), "&%5.3f" % (AUC), 
      "&%8.3f" % (MCC), "&%8.3f" % (AP))
    else:
        print("& %5.3f" % (ACCURACY), " &%7.3f" % (weighted_accuracy), " &%15.3f" % (precision), 
      " &%10.3f" % (SE), " &%5.3f" % (SP), " &%5.3f" % (F1), "&%5.3f" % (AUC), "& ",
        MCC, "&%8.3f" % (AP))

def test():
    smiles = ['O=C(NCCC(O)=O)C(C=C1)=CC=C1/N=N/C(C=C2C(O)=O)=CC=C2OCCOC3=CC=C(NC4=NC=C(C)C(NC5=CC=CC(S(NC(C)(C)C)(=O)=O)=C5)=N4)C=C3', 'OCCOC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(NC(C)(C)C)(=O)=O)=C3)=N2)C=C1']
    smiles.extend(['C1CCC(C1)C(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3', 'CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N'])
    smiles.extend(['CCS(=O)(=O)N1CC(C1)(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3', 'C1CC1C(=O)NC2=NN3C(=N2)C=CC=C3C4=CC=C(C=C4)CN5CCS(=O)(=O)CC5', 'CCC1CN(CC1C2=CN=C3N2C4=C(NC=C4)N=C3)C(=O)NCC(F)(F)F'])
    smiles.extend(['OC(COC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(NC(C)(C)C)(=O)=O)=C3)=N2)C=C1)=O', 'O=C(NCCC(O)=O)C(C=C1)=CC=C1/N=N/C(C=C2C(O)=O)=CC=C2OCCOC3=CC=C(NC4=NC=C(C)C(NC5=CC=CC(S(N)(=O)=O)=C5)=N4)C=C3','OC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(NC(C)(C)C)(=O)=O)=C3)=N2)C=C1', 'OCCOC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(N)(=O)=O)=C3)=N2)C=C1'])
    smiles.extend(['CC1=CN=C(N=C1NC2=CC(=CC=C2)S(=O)(=O)NC(C)(C)C)NC3=CC=C(C=C3)OCCN4CCCC4','C1CCN(C1)CCOC2=C3COCC=CCOCC4=CC(=CC=C4)C5=NC(=NC=C5)NC(=C3)C=C2'])
                
    chemical_names = ['MMT3-72', 'MMT3-72-M2', 'Ruxolitinib', 'Tofacitinib', 'Baricitinib', 'Filgotinib', 'Upadacitinib']    
    chemical_names.extend(['M3', 'M4', 'M5', 'M1', 'Fedratinib', 'Pacritinib'])
    MACCS_list = smile_list_to_MACCS(smiles)
    header = ['bit' + str(i) for i in range(167)]
    df = pd.DataFrame(MACCS_list,columns=header)
    maccs = df.values
    return smiles, chemical_names, maccs

def load_model(ml, enzyme):
    print('Load model ', ml, ' for enzyme ', enzyme)
    if ml == 'knn':
        filename = ml + '_' + enzyme
        model = pickle.load(open('model/'+filename, 'rb'))
        return model
    elif ml!='CNN' and ml!='GVAE' and ml !='ChemBERTa':
        filename = ml + '_' + enzyme + '.sav' 
        model = pickle.load(open('model/'+filename, 'rb'))
        return model
    else: # CNN, GVAE, ChemBERTa
        filename = ml + '_' + enzyme + '.pt'
        if ml == 'GVAE':
            path = ''
        elif ml == 'CNN':
            path = 'CNN_CV/'
        elif ml == 'ChemBERTa':
            path = 'ChemBERT_CV/'
        file_name = path + filename
        model = torch.load('model/'+file_name)
        model.eval()
    return model
    

def get_tpr_fpr(y_test, y_prob):
    try:
        if y_prob.shape[1] == 2:
            proba = y_prob[:, 1]
    except:
        proba = y_prob
    # Get AUC values for AUC figure
    roc_values = []
    for thresh in np.linspace(0, 1, 100):
        preds = get_preds(thresh, proba)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        roc_values.append([tpr, fpr])
    tpr_values, fpr_values = zip(*roc_values)
    return tpr_values, fpr_values

        
def save_tpr_fpr(ml, enzyme, tpr_values, fpr_values): 
    tpr_file = 'AUC/' + ml + '_' + enzyme + '_tpr.pickle'
    fpr_file = 'AUC/' + ml + '_' + enzyme + '_fpr.pickle'
    with open(tpr_file, 'wb') as f:
        pickle.dump(tpr_values, f)
    with open(fpr_file, 'wb') as f:
        pickle.dump(fpr_values, f)  

def load_tpr_fpr(ml, enzyme): 
    tpr_file = 'AUC/' + ml + '_' + enzyme + '_tpr.pickle'
    fpr_file = 'AUC/' + ml + '_' + enzyme + '_fpr.pickle'
    with open(tpr_file, 'rb') as f:
        tpr = pickle.load(f)
    with open(fpr_file, 'rb') as f:
        fpr = pickle.load(f)
    return tpr, fpr


def load_data(enzyme):
    path = 'data/' + enzyme + '_' + 'MACCS.csv'
    data = pd.read_csv(path)
    header = ['bit'+str(i) for i in range(167)]
    X = data[header]
    y = data['Activity']
    return X, y

def load_smi_y(enzyme):
    try:
        path = 'data/' + enzyme + '_' + 'MACCS.csv'
        data = pd.read_csv(path)
    except: 
        path = enzyme + '_' + 'MACCS.csv'
        data = pd.read_csv(path)
    
    X = data['Smiles']
    y = data['Activity']
    return X, y

# Works for SVM, XGBoost
def AUC_singe_test(ml, enzyme, random_state=42): 
    model = load_model(ml, enzyme)

    X, y = load_data(enzyme)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        random_state=random_state, test_size = 0.2)
    y_prob = model.predict_proba(X_test)
    tpr, fpr = get_tpr_fpr(y_test, y_prob)
    title = enzyme + ' Receiver Operating Characteristic Curve' 
    # tpr, fpr = load_tpr_fpr(ml, enzyme)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(fpr, tpr, label=ml)
    ax.plot(np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            label='baseline',
            linestyle='--')
    plt.title(title, fontsize=18)
    plt.ylabel('TPR', fontsize=16)
    plt.xlabel('FPR', fontsize=16)
    plt.legend(fontsize=12)
    # plt.savefig('figures/'+enzyme+'.png')
    plt.show()

