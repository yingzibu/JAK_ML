import pandas as pd
import pubchempy as pcp
from os import walk
import os 
from collections import OrderedDict
import numpy as np

# Input: PUBCHEM_CID, int
# Output: smile of this compound
# example: 
# get_smile(3081361)
def get_smile(cid):
    #smiles = []
    mol = pcp.Compound.from_cid(cid)
    if mol:    
        smi = mol.canonical_smiles   
        return smi
    else:
        return None
        #print(smi)
        
        


# Input: file name, csv file
# Extract "PUBCHEM_CID", "PUBCHEM_ACTIVITY_OUTCOME"
# Replace 'Active' as 1, 'Inactive' as 0, 'Unspecified' as -1, delete 'Inconclusive' 
# Output: panda frame, column 0: 'PUBCHEM_CID', column 1: 'PUBCHEM_ACTIVITY_OUTCOME', column 2: 'Smiles'
# Output: csv file saved in smile_added_pubchem folder

# Example: 
# replace_label('AID_256646.csv')
def replace_label(file, enzyme):
    filename = 'pubchem/' + enzyme +'/' + file
    data = pd.read_csv(filename)
    index1 = data.index[data['PUBCHEM_CID'].isnull() == False].tolist()
    index2 = data.index[data['PUBCHEM_ACTIVITY_OUTCOME']!= 'Inconclusive'].tolist()
    index = list(set(index1) & set(index2))
    new_file = data.loc[index,:][['PUBCHEM_CID', 'PUBCHEM_ACTIVITY_OUTCOME']]
    new_file['PUBCHEM_CID'] = new_file['PUBCHEM_CID'].astype(int)
    
    new_file.replace('Active', 1, inplace=True)
    new_file.replace('Inactive', 0, inplace=True)
    new_file.replace('Unspecified', -1, inplace=True)
    #print(new_file)
    CIDs = new_file['PUBCHEM_CID'].tolist()
    smiles = []
    for (i, CID) in enumerate(CIDs): 
        #print(CID)
        smile = get_smile(CID)
        #print(smile)
        smiles.append(smile)
    new_file['Smiles'] = smiles
    print('saved file path:', 'smile_added_pubchem/'+ enzyme  + '/' + file)
    new_file.to_csv('smile_added_pubchem/'+ enzyme +'/'+ file, index=False) 
    return new_file


    # input: jak category, 
# example:
# handle_files('JAK2')  
# read the pubchem/enzyme folder, all the files extract cid, activity and converted smiles from cid
# save file in 'smile_added_pubchem/'+ enzyme folder, with the same filename as original file
# save a large file in current path, containing all compound ID, activity and smiles
# No output
def handle_files(enzyme): 
    filenames_all = next(walk('pubchem/' + enzyme), (None, None, []))[2]
    initial_len_file = len(filenames_all)
    print('total file number: ', initial_len_file)
    
    filenames_handled = next(walk('smile_added_pubchem/' + enzyme), (None, None, []))[2]
    if len(filenames_handled) != 0:
        print('files already handled, ', len(filenames_handled))
        handled = list(set(filenames_handled) & set(filenames_all))
        print('make sure the number of handled file: ', len(handled))
        unhandled = list(set(filenames_all) - set(handled))
        filenames = unhandled        
    else:
        filenames = filenames_all
    #compound = pd.DataFrame()
    print('Unhandled file number, ', len(filenames))
    while len(filenames) != 0:
        if len(filenames) % 100 == 0:
            print('Still ', len(filenames), ' files left')
        filename = filenames.pop(0)
        try: 
            replace = replace_label(filename, enzyme)
            # compound = pd.concat([compound, replace], ignore_index=True)
        except:
            filenames.insert(0, filename)
    # compound.to_csv(enzyme+'.csv', index=False) 
    print('end ', enzyme)
    #return compound

# Input: enzyme
# EXAMPLE: assay_type('JAK1')
# Determine the assay type of all files in this enzmye folder
# Output: dictionary type, filename: assay_type
# Output: abnormal files not having standard type column, check seperately 
# Output: all assays in this enzyme folder
def assay_type(enzyme): 
    filenames_all = next(walk('pubchem/' + enzyme), (None, None, []))[2]
    assay_types = []
    assays = []
    dicts = {}
    abnormal_files = []
    for filename in filenames_all: 
        try: 
            data = pd.read_csv('pubchem/' + enzyme + '/' + filename)
        except:
            print('error occurs when opening ', 'pubchem/' + enzyme + '/' + filename)
        try:
            assay_type = data['Standard Type'][data['Standard Type'].shape[0]-1]
            assay_types.append(assay_type)
            if assay_type not in assays:
                assays.append(assay_type)
        except:
            print('No standard type column:', filename)
            abnormal_files.append(filename)
            #print(data.head())
    dicts = dict(zip(filenames_all, assay_types))
    return dicts, abnormal_files, assays
    
# Input: dataframe
# Count the number of inhibitors, noninhibitors, unspecified
def count_category(all_data): 
    if 'PUBCHEM_ACTIVITY_OUTCOME' in all_data.columns: 
        labels = all_data['PUBCHEM_ACTIVITY_OUTCOME']
    elif 'Activity' in all_data.columns:
        labels = all_data['Activity'] 
    else:
        print('Cannot find activity column')
        return
    try:
        print('Active inhibitors #: ', labels.value_counts()[1])
    except:
        print('Active inhibitors #: ', 0)
    try: 
        print('Non inhibitors #: ', labels.value_counts()[0])
    except:
        print('Non inhibitors #: ', 0)
    try: 
        print('Unspecified #: ', labels.value_counts()[-1])
    except:
        print('Unspecified #: ', 0)

# Input: certain assay name, enzyme
# extract all files done this assay, compile into a large file
def certain_assay_file(assay_name, enzyme): 
    dictionary, _, assays = assay_type(enzyme)
    if assay_name not in assays:
        print('Assay name is not in any csv file for ', enzyme)
        return
    else: 
        all_data = pd.DataFrame()
        for filename, assay in dictionary.items():
            if assay == assay_name:
                filepath = 'pubchem/' + enzyme + '/' + filename
                
                filepath_smile = 'smile_added_pubchem/'+ enzyme + '/' + filename
                print(assay_name, ' found in ', filepath, ' extract smile info from ', filepath_smile)
                data = pd.read_csv(filepath_smile)
                all_data = pd.concat([all_data, data], ignore_index=True)
        if '/' in assay_name:
            assay_name = assay_name.replace('/', '_')
            
        assay_filepath = 'assay/' + enzyme + '/' + assay_name + '.csv'
        print('csv files of ', assay_name, ' saved in ', assay_filepath)
        
        #print(all_data['PUBCHEM_ACTIVITY_OUTCOME'].value_counts()[1])
        count_category(all_data)
        all_data.to_csv(assay_filepath, index=False)
        #return all_data
                
def check_handled(enzyme):
    files_all = next(walk('pubchem/' + enzyme), (None, None, []))[2]
    files_handled = next(walk('smile_added_pubchem/' + enzyme), (None, None, []))[2]
    if set(files_all) == set(files_handled):
        print('all files are handled')
    else: 
        handled = list(set(files_all) & set(files_handled))
        unhandled = list(set(files_all) - set(handled))
        print('for ', enzyme, ' unhandled file names are ', unhandled)
        check_unhandled = list(set(files_all) - set(files_handled))
        print('double check ', 'for ', enzyme, ' unhandled file names are ', check_unhandled)
        if set(unhandled) != set(check_unhandled):
            print('Discrepancy! ')
        else:
            print('There are ', len(check_unhandled), ' files in enzyme ', enzyme)
            return check_unhandled

# Input: pd frame
# Change the labels of unspecified -1 into non-inhibitor 0
# Output: pd frame
def _1to0(data):
    count_category(data)
    data['PUBCHEM_ACTIVITY_OUTCOME'] = data['PUBCHEM_ACTIVITY_OUTCOME'].replace(-1, 0)
    print('unspecified -1 was changed to noninhibitor 0')
    count_category(data)
    return data

# EXAMPLE:
# _1to0_col_name_change('JAK1')
def _1to0_col_name_change(enzyme): 
    #enzyme = 'JAK1'
    files = next(walk('assay/' + enzyme), (None, None, []))[2]
    for file in files: 
        data = pd.read_csv('assay/' + enzyme + '/' + file)
        data = _1to0(data)
        smiles = data['Smiles'].tolist()
        data.rename(columns={'PUBCHEM_ACTIVITY_OUTCOME':'Activity'}, inplace=True)

        data = delete_duplicate(data)
        
        filepath = 'assay_relabel/' + enzyme + '/' + file
        print('file column renamed and saved as ', filepath)

        data.to_csv(filepath, index=False)

def remove_empty_csv(enzyme):
    path = 'pubchem/' + enzyme
    filenames = next(walk(path), (None, None, []))[2]
    print('Check empty files in ', path, ' folder')
    i = 0
    for file in filenames: 
        filename = 'pubchem/' + enzyme + '/' + file
        if os.stat(filename).st_size == 0:
        #note: file has to be in same directory as python script#
            os.remove(filename)
            print('empty file removed ', filename)
            i = i + 1
    print(i, ' empty files in ', path, ' folder are deleted')

# Input: dataframe, column ['Activity'], ['Smiles']
# Output: dataframe, duplicate deleted  
def delete_duplicate(data):
    new_data = data.drop_duplicates()
    return new_data
def delete_duplicate_smiles(data): 
    new_data = data.drop_duplicates(subset=['Smiles'])
    return new_data
def delete_duplicate_smiles_activity(data): 
    try: 
        if 'Activity' in data.columns:
            new_data = data.drop_duplicates(subset=['Smiles', 'Activity'])
        elif 'PUBCHEM_ACTIVITY_OUTCOME' in  data.columns:
            new_data = data.drop_duplicates(subset=['Smiles', 'PUBCHEM_ACTIVITY_OUTCOME'])
        return new_data
    except:
        print('cannot find acitivity in data')
    

