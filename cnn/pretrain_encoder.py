import pandas as pd
import torch
import os
from tqdm import tqdm
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from encoder import encoder
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import rdkit
from rdkit import Chem
from const import *


def smiles2adj(sml):
    """Argument for the RD2NX function should be a valid SMILES sequence
    returns: the graph
    """
    # atom2num={6:0, 7:1, 8:2, 9:3, 11:4, 14:5, 15:6, 16:7, 17:8, 19:9, 35:10, 53:11, 5:12, 1:13}
    m = rdkit.Chem.MolFromSmiles(sml)
    # m = rdkit.Chem.AddHs(m)
    order_string = {
        rdkit.Chem.rdchem.BondType.SINGLE: 1,
        rdkit.Chem.rdchem.BondType.DOUBLE: 2,
        rdkit.Chem.rdchem.BondType.TRIPLE: 3,
        rdkit.Chem.rdchem.BondType.AROMATIC: 4,
    }
    try:
        N = len(list(m.GetAtoms()))
    except:
        print(sml)

    return rdMolDescriptors.GetMACCSKeysFingerprint(m)

    # adj = np.zeros((adj_max, adj_max))
    # for j in m.GetBonds():
    #     u = min(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
    #     v = max(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
    #     order = j.GetBondType()
    #     if order in order_string:
    #         order = order_string[order]
    #     else:
    #         raise Warning("Ignoring bond order" + order)
    #     if u < 80 and v < 80:
    #         # u=m.GetAtomWithIdx(u).GetAtomicNum()
    #         # v=m.GetAtomWithIdx(v).GetAtomicNum()
    #         # print(u,v)
    #         adj[u, v] = 1#order
    #         # if u!=v:
    #         adj[v, u] = 1#order
    # # nodes[:,1] =adj.sum(axis=0)
    # # adj += np.eye(N)
    # return adj


class pretrain_dataset(Dataset):
    def __init__(self, dataframe, max_len=max_len):
        super(pretrain_dataset, self).__init__()

        self.len = len(dataframe)
        self.dataframe = dataframe
        self.max_len = max_len

    def __getitem__(self, idx):
        X = torch.zeros(self.max_len)
        sml = self.dataframe.canonical_smiles[idx]
        # print(X)
        for idx, atom in enumerate(list(sml)[:self.max_len]):
            X[idx] = vocabulary[atom]
        chem_id = self.dataframe.chembl_id[idx]
        s = self.dataframe.fps[idx]
        s = list(s)
        adj = torch.tensor([int(b) for b in s])
        # adj = torch.tensor(smiles2adj(sml))
        # print(X,adj,chem_id)
        return X, adj, chem_id

    def __len__(self):
        return self.len


if __name__ == '__main__':
    df = pd.read_csv('pretrained_data/chembl_30_chemreps.txt', sep='\t')
    params = {'batch_size': 1024, 'shuffle': True, 'drop_last': False, 'num_workers': 0}
    data = pretrain_dataset(df)
    loader = DataLoader(data, **params)

    e = encoder(max_len, len(vocabulary)).cuda()
    load_epoch = -1
    if load_epoch != -1:
        e.load_state_dict(
            torch.load('checkpoints/CNN_encoder_pretrain' + str(load_epoch) + ".pt", map_location=device))

    optimizer = optim.SGD(params=e.parameters(), lr=0.9, weight_decay=1e-2)
    loss_function = nn.BCEWithLogitsLoss()

    epoch = 20
    loss_list = []
    for i in range(load_epoch + 1, epoch):
        total_loss = 0
        for idx, (X, adj, chem_id) in tqdm(enumerate(loader), total=len(loader)):
            optimizer.zero_grad()
            X = X.cuda().long()
            adj = adj.float().cuda()
            output = e(X)
            # print(X.shape,output.shape)
            loss = loss_function(output, adj)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("epoch ", i, "loss", total_loss)
        loss_list.append(total_loss / len(loader))
        torch.save(e.state_dict(), 'checkpoints/CNN_encoder_pretrain' + str(i) + ".pt")
