import os
from tqdm import tqdm
import math
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from CNNforclassification import CNNforclassification
from smiles2graph import gen_smiles2graph
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from const import *


class jak_dataset(Dataset):
    def __init__(self, dataframe, max_len=80):
        super(jak_dataset, self).__init__()
        self.len = len(dataframe)
        self.dataframe = dataframe
        self.max_len = max_len

    def __getitem__(self, idx):
        X = torch.zeros(self.max_len)
        y = 1 if self.dataframe.Activity[idx] == 1 else 0
        for idx, atom in enumerate(list(self.dataframe.Smiles[idx])[:self.max_len]):
            X[idx] = vocabulary[atom]

        return X.long(), y

    def __len__(self):
        return self.len


if __name__ == '__main__':
    jak_number = int(cuda_num / 2 + 1)

    # jak_df = pd.read_csv('jak/final_JAK'+str(jak_number)+'.csv',index_col=0).reset_index(drop=True)
    jak_df = pd.read_csv('jak/JAK' + str(jak_number) + '_final.csv')
    # jak1_small_df = pd.read_csv('JAK'+str(jak_number)+'_active.csv').drop(columns='Unnamed: 0')
    # jak1_small_df.columns = jak1_df.columns
    # jak1_small_df['Activity'] = 1 - jak1_small_df['Activity']
    # jak_df = pd.concat([jak1_df, jak1_small_df]).reset_index(drop=True).drop_duplicates().reset_index(drop=True)
    print(jak_df)

    weight_dict = {1: torch.tensor([3.0, 1.0]), 2: torch.tensor([2.0, 1.0]), 3: torch.tensor([2.0, 1.0]),
                   4: torch.tensor([2.0, 1.0])}

    k = 5
    kfold = KFold(n_splits=k, shuffle=True)
    params = {'batch_size': 16, 'shuffle': True, 'drop_last': False, 'num_workers': 0}
    epoches = 20

    known_df = pd.DataFrame(known_drugs)
    known_df.columns=['Smiles']
    known_df['Activity']=0
    known_data=jak_dataset(known_df)
    known_loader=DataLoader(known_data,**params)
    print(known_df)

    acc_know_pred = torch.zeros(len(known_drugs), 2)
    all_y_true = []
    all_y_pred = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(jak_df)):

        test_df = jak_df.iloc[test_ids].reset_index(drop=True)
        print(test_df)
        test_data = jak_dataset(test_df)
        test_loader = DataLoader(test_data, **params)
        train_df = jak_df.iloc[train_ids].reset_index(drop=True)

        train_data = jak_dataset(train_df)
        train_loader = DataLoader(train_data, **params)
        c = CNNforclassification(max_len, len(vocabulary)).cuda()
        optimizer = optim.SGD(params=c.parameters(), lr=0.1, weight_decay=1e-2)
        loss_function = nn.CrossEntropyLoss(weight=weight_dict[jak_number].cuda())
        c.train()
        for epoch in range(epoches):
            print("EPOCH -- {}".format(epoch))
            total_loss = 0
            for idx, (x, y_true) in tqdm(enumerate(train_loader), total=len(train_loader)):
                # print(y_true)
                optimizer.zero_grad()
                if cuda_available:
                    x=x.cuda()
                    y_true=y_true.cuda()
                output = c(x)
                loss = loss_function(output, y_true)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        torch.save(c.state_dict(),
                   'checkpoints/cnn_for_classification_jak_' + str(jak_number) + '_fold_' + str(fold) + ".pt")

        c.eval()
        accumulate_y_pred = []
        accumulate_y_true = []
        for idx, (X, y_true) in tqdm(enumerate(test_loader), total=len(test_loader)):
            output = c(X.cuda())
            _, y_pred = torch.max(output, 1)
            accumulate_y_pred.extend(y_pred.tolist())
            accumulate_y_true.extend(y_true.tolist())
        all_y_pred.extend(accumulate_y_pred)
        all_y_true.extend(accumulate_y_true)

        print(
            classification_report(accumulate_y_true, accumulate_y_pred, target_names=['active', 'inactive'], digits=3))

        for idx, (X, y_true) in tqdm(enumerate(known_loader), total=len(known_loader)):
            output = c(X.cuda())
            _, y_pred = torch.max(output, 1)
            print(torch.max(torch.softmax(output, 1), 1)[0].tolist())
            print(y_pred.tolist())

        for idx, bit in enumerate(y_pred.tolist()):
            acc_know_pred[idx][bit] += 1
    print(torch.max(acc_know_pred, 1)[1])
    tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()
    print("accuracy:", (tp + tn) / (tn + fp + fn + tp))
    print("weighted accuracy:", (tp / (fn + tp) + tn / (tn + fp)) / 2)
    print("precision:", tp / (fp + tp))
    print("recall:", tp / (fn + tp))
    print("neg recall:", tn / (tn + fp))
    print("F1:", tp / (tp + 0.5 * (fp + fn)))
    print("AUC", roc_auc_score(all_y_true, all_y_pred, average='macro'))
    print("MCC:", (tp * tn - fn * fp) / math.sqrt((tp + fp) * (tp + fn) * (tn + fn) * (tn + fp)))
    print("AP:", average_precision_score(all_y_true, all_y_pred, average='macro', pos_label=1))