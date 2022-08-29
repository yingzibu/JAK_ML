from function import load_model, test, smile_list_to_MACCS, load_tpr_fpr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

for enzyme in ['JAK1', 'JAK2', 'JAK3', 'TYK2']:
    title = enzyme + ' Receiver Operating Characteristic Curve' 
    models = ['SVM_linear', 'SVM_poly', 'SVM_rbf', 'SVM_sigmoid', 'RF', 'XGBoost', 'CNN', 'GVAE']
    fig, ax = plt.subplots(figsize=(10,10))
    for ml in models:
        tpr, fpr = load_tpr_fpr(ml, enzyme)
        ax.plot(fpr, tpr, label=ml)
    ax.plot(np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            label='baseline',
            linestyle='--')
    plt.title(title, fontsize=18)
    plt.ylabel('TPR', fontsize=16)
    plt.xlabel('FPR', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig('figures/'+enzyme+'.png')
    plt.show()