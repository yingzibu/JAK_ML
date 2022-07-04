import ssl
import certifi
import pandas as pd 
import pubchempy as pcp
from os import walk
import os
from collections import OrderedDict
import numpy as np
import schedule
import time 
import matplotlib.pyplot as plt 
from functions import delete_duplicate, get_smile, replace_label, handle_files, \
                      assay_type, count_category, certain_assay_file, \
                      check_handled, _1to0, _1to0_col_name_change, \
                      remove_empty_csv, delete_duplicate, delete_duplicate_smiles, \
                      delete_duplicate_smiles_activity

ssl._create_default_https_context = ssl._create_unverified_context
def timed_job():
    handle_files('FLT3')
    print('this round of time finished?')
schedule.every(2).minutes.until('2022-07-04 13:15').do(timed_job)
while 1:
    schedule.run_pending()
    time.sleep(1)







# maccs = pd.read_csv('playground/JAK1_MACCS.csv')
# maccs.head()
# maccs = maccs.drop(['Unnamed: 0'], axis=1)
# maccs.head()
# y = pd.read_csv('playground/JAK1_all.csv')
# label = y['Activity']
# from KNN import KNN_batch
# KNN_batch(maccs, label)

#handle_files('FLT3')
# data = pd.read_csv('assay/test/Control.csv')
# print(data)
# count_category(data)
# new_data = delete_duplicate(data)
# count_category(new_data)
# another_data = delete_duplicate_smiles(data)
# count_category(another_data)

# a_data = delete_duplicate_smiles_activity(data)
# count_category(a_data)
# enzyme ='JAK3'
# dict, abn, assays = assay_type(enzyme)
# #print(abnormal_files)
# print(assays)


# _1to0_col_name_change('TYK2')
# data = pd.read_csv('assay_relabel/' + enzyme + '/' + 'IC50.csv')
# print(data.head())

# data.rename(columns={'PUBCHEM_ACTIVITY_OUTCOME':'Activity'}, inplace=True)
# print(data.head())



#dicts, abnormal_files, assays = assay_type('TYK2')

#print(assays)
#print(abnormal_files)
# for assay in assays: 
#     certain_assay_file(assay, 'TYK2')

# for i in ['JAK1', 'JAK2', 'JAK3', 'TYK2']:
#     try:
#         files = check_handled(i)
#         if files == None:
#             print('for enzyme ', i)
#             print('All files are handled')
        
#         print('-------------')
#         for j in files: 
#             filepath = 'pubchem/'+i+'/'+j
#             print(filepath)
#             #data = pd.read_csv(filepath)
#             #count_category(data)
#             #replace_label(j, i)
#     except:
#         print('for enzyme ', i)
#         print('no output')
#         print('--------------')