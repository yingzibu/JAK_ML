import pandas as pd
import torch
import os
from tqdm import tqdm
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from bert_encoder import chembert_encoder
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from const import *

class pretrain_dataset(Dataset):
    def __init__(self, dataframe, max_len=max_len):
        super(pretrain_dataset, self).__init__()

        self.len = len(dataframe)
        self.dataframe = dataframe
        self.max_len = max_len

    def __getitem__(self, idx):
        sml = self.dataframe.canonical_smiles[idx]
        # print(X)
        chem_id = self.dataframe.chembl_id[idx]
        s = self.dataframe.fps[idx]
        s = list(s)
        adj = torch.tensor([int(b) for b in s])
        return sml, adj, chem_id

    def __len__(self):
        return self.len

if __name__ == '__main__':
    df = pd.read_csv('pretrained_data/chembl_30_chemreps.txt', sep='\t')
    params = {'batch_size': 16, 'shuffle': True, 'drop_last': False, 'num_workers': 0}
    data = pretrain_dataset(df)
    loader = DataLoader(data, **params)

    model=chembert_encoder()
    if cuda_available:
        model=model.cuda()
    load_epoch = -1
    if load_epoch != -1:
        model.load_state_dict(
            torch.load('checkpoints/chem_bert_encoder_pretrain_' + str(load_epoch) + ".pt", map_location=device))

    optimizer = optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_function = nn.MSELoss()

    epoch = 30
    loss_list = []
    for i in range(load_epoch + 1, epoch):
        total_loss = 0
        for idx, (x, adj, chem_id) in tqdm(enumerate(loader), total=len(loader)):
            optimizer.zero_grad()
            adj = adj.float().cuda()
            output = model(list(x))
            loss = loss_function(output, adj)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("epoch ", i, "loss", total_loss)
        loss_list.append(total_loss / len(loader))
        torch.save(model.state_dict(), 'checkpoints/chem_bert_encoder_pretrain_' + str(i) + ".pt")
