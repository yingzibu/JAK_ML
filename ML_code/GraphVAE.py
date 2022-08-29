from pickletools import float8
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.utils.data 
from torch_geometric.data import DataLoader
from torch_geometric.data import Data

from torch_geometric.nn import GATConv, RGCNConv, GCNConv, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, roc_auc_score

import rdkit
from rdkit.Chem.Scaffolds import MurckoScaffold

from itertools import compress
import random
from collections import defaultdict

import pickle

rootpath = 'Downloads/'
device = 'cpu'

def gen_smiles2graph(sml):
    """Argument for the RD2NX function should be a valid SMILES sequence
    returns: the graph
    """
    ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 30, 33, 34, 35, 36, 37, 38, 47, 52, 53, 54, 55, 56, 83, 88]
    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i
    m = rdkit.Chem.MolFromSmiles(sml)
#    m = rdkit.Chem.AddHs(m)
    order_string = {
        rdkit.Chem.rdchem.BondType.SINGLE: 1,
        rdkit.Chem.rdchem.BondType.DOUBLE: 2,
        rdkit.Chem.rdchem.BondType.TRIPLE: 3,
        rdkit.Chem.rdchem.BondType.AROMATIC: 4,
    }
    N = len(list(m.GetAtoms()))
    nodes = np.zeros((N, 6))

    try:
        test = m.GetAtoms()
    except:
        return 'error', 'error', 'error'

    for i in m.GetAtoms():
        atom_types= dic[i.GetAtomicNum()]
        atom_degree= i.GetDegree()
        atom_form_charge= i.GetFormalCharge()
        atom_hybridization= i.GetHybridization()
        atom_aromatic= i.GetIsAromatic()
        atom_chirality= i.GetChiralTag()
        nodes[i.GetIdx()] = [atom_types, atom_degree, atom_form_charge, atom_hybridization, atom_aromatic, atom_chirality]

    adj = np.zeros((N, N))
    orders = np.zeros((N, N))
    for j in m.GetBonds():
        u = min(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        v = max(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        order = j.GetBondType()
        if order in order_string:
            order = order_string[order]        
        else:
            raise Warning("Ignoring bond order" + order)
        adj[u, v] = 1
        adj[v, u] = 1
        orders[u, v] = order
        orders[v, u] = order
#    adj += np.eye(N)
    return nodes, adj, orders

def preprocess():
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    global rootpath
    rootpath = '/Users/uranaieiko_1/Desktop/ML_JAKi_20220612/new_data/final_data_20220721/delete_controversy/data/'
    JAK = [rootpath+'TYK2_final.csv']
    LABEL = [rootpath+'JAK1_graph_labels.txt']
    nodes = []
    edges = []
    relations = []
    y = []
    smiles_all = []
    for path1, path2 in zip(JAK, LABEL):
        smiles = pd.read_csv(path1)['Smiles'].tolist()
#         from collections import OrderedDict
#         d = OrderedDict((x, smiles.index(x)) for x in smiles)
#         smiles = list(d.keys())
#         index = list(d.values())
        y_ = ((np.array(pd.read_csv(path1)['Activity'].tolist())>0)*1).tolist()
#         y_ = list(map(y_.__getitem__, index))
        smiles_all.extend(smiles)
        lens = []
        adjs = []
        ords = []
        for i in range(len(smiles)):
            node, adj, order = gen_smiles2graph(smiles[i])
            if node == 'error':
                continue
            lens.append(adj.shape[0])
            adjs.append(adj)  
            ords.append(order)
            node[:,2] += 1
            node[:,3] -= 1
            nodes.append(node)
        adjs = np.array(adjs)
        lens = np.array(lens)
    
        def file2array(path, delimiter=' '):     
            fp = open(path, 'r', encoding='utf-8')
            string = fp.read()              
            fp.close()
            row_list = string.splitlines()  
            data_list = [[float(i) for i in row.strip().split(',')] for row in row_list]
            return np.array(data_list)
    
        def adj2idx(adj):
            idx = []
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i,j] == 1:
                        idx.append([i,j])
            return np.array(idx)
        
        def order2relation(adj):
            idx = []
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i,j] != 0:
                        idx.extend([adj[i,j]])
            return np.array(idx)
        
        for i in range(lens.shape[0]):
            adj = adjs[i]
            order = ords[i]
            idx = adj2idx(adj)
            relation = order2relation(order)-1
            edges.append(idx)
            relations.append(relation)
    
        y.append(np.array(y_))
        
#    nodes = np.array(nodes)
#    edges = np.array(edges)
    y  = np.concatenate(y)
    y = np.array(y) 
    
    return smiles_all, nodes, edges, relations, y

def preprocess_pretrain(idx, start, end):
    f = open(rootpath+'chembl_30_chemreps.txt',"r")
    idx = idx[start: end]+1
    lines = list(map(f.readlines().__getitem__, idx))
    smiles = []
    for x in lines:
        smiles.append(x.split('\t')[1])
    f.close()

    nodes = []
    edges = []
    relations = []
    lens = []
    adjs = []
    ords = []
    for i in range(len(smiles)):
        node, adj, order = gen_smiles2graph(smiles[i])
        if node == 'error':
            print(i, smiles, 'error')
            continue
        lens.append(adj.shape[0])
        adjs.append(adj)  
        ords.append(order)
        node[:,2] += 1
        node[:,3] -= 1
        nodes.append(node)
        if i%1000 == 0:
            print('molecule# ', i)

    adjs = np.array(adjs)
    lens = np.array(lens)
    
    def file2array(path, delimiter=' '):     
        fp = open(path, 'r', encoding='utf-8')
        string = fp.read()              
        fp.close()
        row_list = string.splitlines()  
        data_list = [[float(i) for i in row.strip().split(',')] for row in row_list]
        return np.array(data_list)

    def adj2idx(adj):
        idx = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i,j] == 1:
                    idx.append([i,j])
        return np.array(idx)
    
    def order2relation(adj):
        idx = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i,j] != 0:
                    idx.extend([adj[i,j]])
        return np.array(idx)
    
    for i in range(lens.shape[0]):
        adj = adjs[i]
        order = ords[i]
        idx = adj2idx(adj)
        relation = order2relation(order)-1
        edges.append(idx)
        relations.append(relation)
    
    return smiles, nodes, edges, relations

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )
    return scaffold

def random_scaffold_split(
    dataset,
    smiles_list,
    task_idx=None,
    null_value=0,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=42,
):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/\
        chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(list(scaffolds.values()))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    return train_idx, valid_idx, test_idx

class GDataset(Dataset):
    def __init__(self, nodes, edges, relations, y, idx):
        super(GDataset, self).__init__()
        
        self.nodes = nodes
        self.edges = edges
        self.y = y
        self.relations = relations
        self.idx = idx
        
    def __getitem__(self, idx):
        idx = self.idx[idx]
        edge_index = torch.tensor(self.edges[idx].T, dtype=torch.long)
        x = torch.tensor(self.nodes[idx], dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.float)
        edge_type = torch.tensor(self.relations[idx], dtype=torch.float)
        return Data(x=x,edge_index=edge_index,edge_type=edge_type,y=y)
    
    def __len__(self):
        return len(self.idx)
    
    def collate_fn(self,batch):
        pass

def weights_init(m): 
    if isinstance(m, (nn.Linear)):                                 
        nn.init.xavier_uniform_(m.weight)
        
class RGCN_VAE(torch.nn.Module):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(RGCN_VAE, self).__init__()
        self.embedding = nn.ModuleList([nn.Embedding(35,in_embd), nn.Embedding(10,in_embd), \
                          nn.Embedding(5,in_embd), nn.Embedding(7,in_embd), \
                          nn.Embedding(5,in_embd), nn.Embedding(5,in_embd)])
        
        self.GATConv1 = RGCNConv(6*in_embd, layer_embd, num_relations)
        self.GATConv2 = RGCNConv(layer_embd, out_embd*2, num_relations)
        
#         self.GATConv1 = GCNConv(6*in_embd, layer_embd, num_relations)
#         self.GATConv2 = GCNConv(layer_embd, out_embd*2, num_relations)
        
        self.GATConv1.reset_parameters()
        self.GATConv2.reset_parameters()
        
        self.activation = nn.Sigmoid()
        self.d = out_embd        
        
        self.pool = GlobalAttention(gate_nn=nn.Sequential( \
                nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)))
        
        self.graph_linear = nn.Linear(out_embd, 1)
        
    def recognition_model(self, x, edge_index, edge_type, batch):
        for i in range(6):
            embds = self.embedding[i](x[:,i])
            if i == 0:
                x_ = embds
            else:
                x_ = torch.cat((x_, embds), 1)
        out = self.activation(self.GATConv1(x_, edge_index, edge_type))
        out = self.activation(self.GATConv2(out, edge_index, edge_type))  
        
#         out = self.activation(self.GATConv1(x_, edge_index))
#         out = self.activation(self.GATConv2(out, edge_index))  
        
        mu = out[:,0:self.d] 
        logvar = out[:,self.d:2*self.d]
        
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        
        return eps.mul(std) + mu

    def generation_model(self, Z): 
        out = self.activation(Z@Z.T)
        
        return out
      
    def forward(self, x, edge_index, edge_type, batch, type_):
        if type_=='pretrain':
            mu, logvar = self.recognition_model(x, edge_index, edge_type, batch)
            Z = self.reparametrize(mu, logvar)
            A_hat = self.generation_model(Z)

            N = x.size(0)
            A = torch.zeros((N,N), device=device)
            with torch.no_grad():
                for i in range(edge_index.size(1)):
                    A[edge_index[0,i], edge_index[1,i]] = 1
          # print(A.size(),A_hat.size())
            return A, A_hat, mu, logvar
        else:
            mu = self.cal_mu(x, edge_index, edge_type, batch)
            out = self.pool(mu, batch)
            out = self.graph_linear(out)
            out = self.activation(out)
            return out
  
    def cal_mu(self, x, edge_index, edge_type, batch):
        mu, _ = self.recognition_model(x, edge_index, edge_type, batch)
        
        return mu
    
class GCN_VAE(torch.nn.Module):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(GCN_VAE, self).__init__()
        self.embedding = nn.ModuleList([nn.Embedding(35,in_embd), nn.Embedding(10,in_embd), \
                          nn.Embedding(5,in_embd), nn.Embedding(7,in_embd), \
                          nn.Embedding(5,in_embd), nn.Embedding(5,in_embd)])
        
        self.GATConv1 = GCNConv(6*in_embd, layer_embd, num_relations)
        self.GATConv2 = GCNConv(layer_embd, out_embd*2, num_relations)
        
        self.GATConv1.reset_parameters()
        self.GATConv2.reset_parameters()
        
        self.activation = nn.Sigmoid()
        self.d = out_embd        
        
        self.pool = GlobalAttention(gate_nn=nn.Sequential( \
                nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)))
        
        self.graph_linear = nn.Linear(out_embd, 1)
        
    def recognition_model(self, x, edge_index, edge_type, batch):
        for i in range(6):
            embds = self.embedding[i](x[:,i])
            if i == 0:
                x_ = embds
            else:
                x_ = torch.cat((x_, embds), 1)
        
        out = self.activation(self.GATConv1(x_, edge_index))
        out = self.activation(self.GATConv2(out, edge_index))  
        
        mu = out[:,0:self.d] 
        logvar = out[:,self.d:2*self.d]
        
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        
        return eps.mul(std) + mu

    def generation_model(self, Z): 
        out = self.activation(Z@Z.T)
        
        return out
      
    def forward(self, x, edge_index, edge_type, batch, type_):
        if type_=='pretrain':
            mu, logvar = self.recognition_model(x, edge_index, edge_type, batch)
            Z = self.reparametrize(mu, logvar)
            A_hat = self.generation_model(Z)

            N = x.size(0)
            A = torch.zeros((N,N), device=device)
            with torch.no_grad():
                for i in range(edge_index.size(1)):
                    A[edge_index[0,i], edge_index[1,i]] = 1
          # print(A.size(),A_hat.size())
            return A, A_hat, mu, logvar
        else:
            mu = self.cal_mu(x, edge_index, edge_type, batch)
            out = self.pool(mu, batch)
            out = self.graph_linear(out)
            out = self.activation(out)
            return out
  
    def cal_mu(self, x, edge_index, edge_type, batch):
        mu, _ = self.recognition_model(x, edge_index, edge_type, batch)
        
        return mu
    
class GAT_VAE(torch.nn.Module):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(GAT_VAE, self).__init__()
        self.embedding = nn.ModuleList([nn.Embedding(35,in_embd), nn.Embedding(10,in_embd), \
                          nn.Embedding(5,in_embd), nn.Embedding(7,in_embd), \
                          nn.Embedding(5,in_embd), nn.Embedding(5,in_embd)])
        
        self.GATConv1 = GATConv(6*in_embd, layer_embd, num_relations)
        self.GATConv2 = GATConv(layer_embd, out_embd*2, num_relations)
        
        self.GATConv1.reset_parameters()
        self.GATConv2.reset_parameters()
        
        self.activation = nn.Sigmoid()
        self.d = out_embd        
        
        self.pool = GlobalAttention(gate_nn=nn.Sequential( \
                nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)))
        
        self.graph_linear = nn.Linear(out_embd, 1)
        
    def recognition_model(self, x, edge_index, edge_type, batch):
        for i in range(6):
            embds = self.embedding[i](x[:,i])
            if i == 0:
                x_ = embds
            else:
                x_ = torch.cat((x_, embds), 1)
        
        out = self.activation(self.GATConv1(x_, edge_index))
        out = self.activation(self.GATConv2(out, edge_index))  
        
        mu = out[:,0:self.d] 
        logvar = out[:,self.d:2*self.d]
        
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        
        return eps.mul(std) + mu

    def generation_model(self, Z): 
        out = self.activation(Z@Z.T)
        
        return out
      
    def forward(self, x, edge_index, edge_type, batch, type_):
        if type_=='pretrain':
            mu, logvar = self.recognition_model(x, edge_index, edge_type, batch)
            Z = self.reparametrize(mu, logvar)
            A_hat = self.generation_model(Z)

            N = x.size(0)
            A = torch.zeros((N,N), device=device)
            with torch.no_grad():
                for i in range(edge_index.size(1)):
                    A[edge_index[0,i], edge_index[1,i]] = 1
          # print(A.size(),A_hat.size())
            return A, A_hat, mu, logvar
        else:
            mu = self.cal_mu(x, edge_index, edge_type, batch)
            out = self.pool(mu, batch)
            out = self.graph_linear(out)
            out = self.activation(out)
            return out
  
    def cal_mu(self, x, edge_index, edge_type, batch):
        mu, _ = self.recognition_model(x, edge_index, edge_type, batch)
        
        return mu
  
class RGCN(torch.nn.Module):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(RGCN, self).__init__()
        self.embedding = nn.ModuleList([nn.Embedding(33,in_embd), nn.Embedding(5,in_embd), \
                          nn.Embedding(3,in_embd), nn.Embedding(4,in_embd), \
                          nn.Embedding(2,in_embd), nn.Embedding(3,in_embd)])
        
        self.RGCNConv1 = RGCNConv(6*in_embd, layer_embd, num_relations)
        self.RGCNConv2 = RGCNConv(layer_embd, out_embd, num_relations)
        self.RGCNConv1.reset_parameters()
        self.RGCNConv2.reset_parameters()
        
        self.activation = nn.Sigmoid()
        
        self.pool = GlobalAttention(gate_nn=nn.Sequential( \
                nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)))
        
        self.graph_linear = nn.Linear(out_embd, 1)
        
    def forward(self, x, edge_index, edge_type, batch):
      for i in range(6):
          embds = self.embedding[i](x[:,i])
          if i == 0:
              x_ = embds
          else:
              x_ = torch.cat((x_, embds), 1)
      
      out = self.activation(self.RGCNConv1(x_, edge_index, edge_type))
      out = self.activation(self.RGCNConv2(out, edge_index, edge_type))  
      out = self.pool(out, batch)
      out = self.graph_linear(out)
      out = F.sigmoid(out)
      return out
  

class GCN(torch.nn.Module):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(GCN, self).__init__()
        self.embedding = [nn.Embedding(33,in_embd), nn.Embedding(5,in_embd), \
                          nn.Embedding(3,in_embd), nn.Embedding(4,in_embd), \
                          nn.Embedding(2,in_embd), nn.Embedding(3,in_embd)]
        
        self.GCNConv1 = GCNConv(6*in_embd, layer_embd)
        self.GCNConv2 = GCNConv(layer_embd, out_embd)
        self.GCNConv1.reset_parameters()
        self.GCNConv2.reset_parameters()
        
        self.activation = nn.Sigmoid()
        
        self.pool = GlobalAttention(gate_nn=nn.Sequential( \
                nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)))
        
        self.graph_linear = nn.Linear(out_embd, 1)
        
    def forward(self, x, edge_index, edge_type, batch):
      for i in range(6):
          embds = self.embedding[i](x[:,i])
          if i == 0:
              x_ = embds
          else:
              x_ = torch.cat((x_, embds), 1)
      
      out = self.activation(self.GCNConv1(x_, edge_index))
      out = self.activation(self.GCNConv2(out, edge_index))  
      out = self.pool(out, batch)
      out = self.graph_linear(out)
      out = F.sigmoid(out)
      return out
  

class GAT(torch.nn.Module):
    def __init__(self, in_embd, layer_embd, out_embd, num_relations, dropout):
        super(GAT, self).__init__()
        self.embedding = [nn.Embedding(33,in_embd), nn.Embedding(5,in_embd), \
                          nn.Embedding(3,in_embd), nn.Embedding(4,in_embd), \
                          nn.Embedding(2,in_embd), nn.Embedding(3,in_embd)]
        
        self.GATConv1 = GATConv(6*in_embd, layer_embd, concat=False, heads=2)
        self.GATConv2 = GATConv(layer_embd, out_embd, concat=False, heads=4)
        self.GATConv1.reset_parameters()
        self.GATConv2.reset_parameters()
        
        self.activation = nn.Sigmoid()
        
        self.pool = GlobalAttention(gate_nn=nn.Sequential( \
                nn.Linear(out_embd, out_embd), nn.BatchNorm1d(out_embd), nn.ReLU(), nn.Linear(out_embd, 1)))
        
        self.graph_linear = nn.Linear(out_embd, 1)
        
    def forward(self, x, edge_index, edge_type, batch):
        for i in range(6):
            embds = self.embedding[i](x[:,i])
            if i == 0:
                x_ = embds
            else:
                x_ = torch.cat((x_, embds), 1)
      
        out = self.activation(self.GATConv1(x_, edge_index))
        out = self.activation(self.GATConv2(out, edge_index))
        out = global_add_pool(out, batch)
        out = self.graph_linear(out)
        out = F.sigmoid(out)
        return out

def loss_function(A, A_hat, mu, logvar):
    A = A.reshape(-1,1).squeeze()
    A_hat = A_hat.reshape(-1,1).squeeze()
    BCE = F.binary_cross_entropy(A_hat, A, reduction='mean')
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    loss = BCE + KLD
    
    return loss

def weighted_BCE(output, target, weight=None):  
    if weight is not None:     
        loss = 1 * (target * torch.log(output)) + \
               weight * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))
    
def pretrain(model, EPOCH, split, batch_size, type_, idx=None, start=0, end=10000):
    if type_ == 'JAK':
        smiles, nodes, edges, relations, y = preprocess()

      # with open('db.pkl', 'wb') as f:
      #     pickle.dump([smiles, nodes, edges, relations], f)
      #     files.download('db.pkl')

    else:
        smiles, nodes, edges, relations = preprocess_pretrain(idx, start, end)
        y = [0]*len(nodes)
    print(len(y), sum(y))
    # train_idx, _, test_idx = random_scaffold_split(y,smiles,frac_train=0.8,frac_valid=0.1,frac_test=0.1)
    
    # train_num = len(train_idx)
    # test_num = len(test_idx)
    
    # train_set = GDataset(nodes, edges, relations, y, train_idx)
    # test_set = GDataset(nodes, edges, relations, y, test_idx)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=test_num, shuffle=False)
    
    
    dataset = GDataset(nodes, edges, relations, y, range(0, len(nodes)))
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model =='RGCN':
        model = RGCN_VAE(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)
        model.apply(weights_init)
    elif model =='GCN':
        model = GCN_VAE(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)
        model.apply(weights_init)
    elif model =='GAT':
        model = GAT_VAE(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)
        model.apply(weights_init)
        
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    for epoch in range(EPOCH):
        print('epoch {}:'.format(epoch+1))
        loss = 0
        for data in train_loader:
            data.to(device)

            A, A_hat, mu, logvar = model(data.x, data.edge_index, data.edge_type, data.batch, 'pretrain')

            loss = loss_function(A, A_hat, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss += loss*A.size(0)
            
        print('\ttrain loss = {}'.format(loss.item()/len(nodes)))
        
    return model
        

def finetune(GraphVAE, EPOCH, split, batch_size, weight):
    smiles, nodes, edges, relations, y = preprocess()
 
    # train_idx, val_idx, test_idx = random_scaffold_split(y,smiles,frac_train=0.8,frac_valid=0.1,frac_test=0.1)
    
    # train_num = len(train_idx)
    # val_num = len(val_idx)
    # test_num = len(test_idx)
    
    # train_set = GDataset(nodes, edges, relations, y, train_idx)
    # val_set = GDataset(nodes, edges, relations, y, val_idx)
    # test_set = GDataset(nodes, edges, relations, y, test_idx)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=test_num, shuffle=False)
    # test_loader = DataLoader(test_set, batch_size=test_num, shuffle=False)
    
    dataset = GDataset(nodes, edges, relations, y, range(0, len(smiles)))
    
    train_num = int(split*len(smiles))
    test_num = len(smiles) - train_num

    train_set, test_set = torch.utils.data.random_split(dataset,[train_num, test_num], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_num, shuffle=False)
    
    optimizer = torch.optim.Adam(GraphVAE.parameters(),lr=1e-4)
    
    weight = torch.FloatTensor([weight])
    
    max_acc = 0
    max_epoch = -1
    
    for epoch in range(EPOCH):
        print('epoch {}:'.format(epoch+1))
        preds = []
        labels = []
        for data in train_loader:
            data.to(device)
            pred = GraphVAE(data.x, data.edge_index, data.edge_type, data.batch, 'finetune')
            loss = weighted_BCE(pred.squeeze(), data.y, weight)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            preds.append((pred.detach().numpy()))
            labels.append(data.y.detach().numpy())
        
        preds  = np.concatenate(preds)
        preds = np.array(preds)
        labels  = np.concatenate(labels)
        labels = np.array(labels)
            
        comp = np.concatenate((((preds>0.5)*1).reshape(-1,1),labels.reshape(-1,1)),1)

        train_0 = sum((comp[:,0]==0) * (comp[:,1]==0)*1) / sum((comp[:,1]==0)*1)
        train_1 = sum((comp[:,0]==1) * (comp[:,1]==1)*1) / sum((comp[:,1]==1)*1)
            
#        print('\ttrain acc = {}\tap = {}\tf1 = {}\troc = {}'.format(round(accuracy_score(labels, (preds>0.5)*1)*100, 2), \
#              round(average_precision_score(labels, preds)*100, 2), round(f1_score(labels, (preds>0.5)*1)*100, 2), round(roc_auc_score(labels, preds)*100, 2)))
        print(train_0, train_1, (train_0+train_1)/2)
        
#         for data in val_loader:
#             data.to(device)
#             preds = model(data.x, data.edge_index, data.edge_type, data.batch).detach().numpy()
#             labels = data.y.detach().numpy()
        
#         comp = np.concatenate((((preds>0.5)*1).reshape(-1,1),labels.reshape(-1,1)),1)

#         val_0 = sum((comp[:,0]==0) * (comp[:,1]==0)*1) / sum((comp[:,1]==0)*1)
#         val_1 = sum((comp[:,0]==1) * (comp[:,1]==1)*1) / sum((comp[:,1]==1)*1)
        
#        print('\ttest acc = {}\tap = {}\tf1 = {}\troc = {}'.format(round(accuracy_score(labels, (preds>0.5)*1)*100, 2), \
#              round(average_precision_score(labels, preds)*100, 2), round(f1_score(labels, (preds>0.5)*1)*100, 2), round(roc_auc_score(labels, preds)*100, 2)))
#         print(val_0, val_1, (val_0+val_1)/2)
        
        for data in test_loader:
            data.to(device)
            preds = GraphVAE(data.x, data.edge_index, data.edge_type, data.batch, 'finetune').detach().numpy()
            labels = data.y.detach().numpy()
        
        comp = np.concatenate((((preds>0.5)*1).reshape(-1,1),labels.reshape(-1,1)),1)

        test_0 = sum((comp[:,0]==0) * (comp[:,1]==0)*1) / sum((comp[:,1]==0)*1)
        test_1 = sum((comp[:,0]==1) * (comp[:,1]==1)*1) / sum((comp[:,1]==1)*1)
        
#        print('\ttest acc = {}\tap = {}\tf1 = {}\troc = {}'.format(round(accuracy_score(labels, (preds>0.5)*1)*100, 2), \
#              round(average_precision_score(labels, preds)*100, 2), round(f1_score(labels, (preds>0.5)*1)*100, 2), round(roc_auc_score(labels, preds)*100, 2)))
        print(test_0, test_1, (test_0+test_1)/2)
        
#         if max_acc <= (val_0+val_1)/2:
#             max_acc = (val_0+val_1)/2
#             max_epoch = epoch + 1
#     print(max_epoch)
    return GraphVAE

def train(model_name, EPOCH, split, batch_size, weight):
    smiles, nodes, edges, relations, y = preprocess()
 
    train_idx, val_idx, test_idx = random_scaffold_split(y,smiles,frac_train=0.8,frac_valid=0.1,frac_test=0.1)
    
    train_num = len(train_idx)
    val_num = len(val_idx)
    test_num = len(test_idx)
    
    train_set = GDataset(nodes, edges, relations, y, train_idx)
    val_set = GDataset(nodes, edges, relations, y, val_idx)
    test_set = GDataset(nodes, edges, relations, y, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=test_num, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=test_num, shuffle=False)
    
#    idx = range(0, len(smiles))
#    
#    dataset = GDataset(nodes, edges, relations, y, idx)
#    
#    train_num = int(split*len(smiles))
#    test_num = len(smiles) - train_num
#
#    train_set, test_set = torch.utils.data.random_split(dataset,[train_num, test_num], generator=torch.Generator().manual_seed(42))
#    
#    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#    test_loader = DataLoader(test_set, batch_size=test_num, shuffle=False)

    if model_name=='RGCN':
        model = RGCN(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)
    elif model_name=='GCN':
        model = GCN(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)
    elif model_name=='GAT':
        model = GAT(in_embd=4, layer_embd=64, out_embd=128, num_relations=4, dropout=0.0)
    model.apply(weights_init)

    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    
    weights = torch.FloatTensor([weight, 1])
    
    max_acc = 0
    max_epoch = -1
    for epoch in range(EPOCH):
        print('epoch {}:'.format(epoch+1))
        preds = []
        labels = []
        for data in train_loader:
            data.to(device)
            pred = model(data.x, data.edge_index, data.edge_type, data.batch)
            loss = weighted_BCE(pred.squeeze(), data.y, weights)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            preds.append((pred.detach().numpy()))
            labels.append(data.y.detach().numpy())
        
        preds  = np.concatenate(preds)
        preds = np.array(preds)
        labels  = np.concatenate(labels)
        labels = np.array(labels)
        
        comp = np.concatenate((((preds>0.5)*1).reshape(-1,1),labels.reshape(-1,1)),1)

        train_0 = sum((comp[:,0]==0) * (comp[:,1]==0)*1) / sum((comp[:,1]==0)*1)
        train_1 = sum((comp[:,0]==1) * (comp[:,1]==1)*1) / sum((comp[:,1]==1)*1)
            
        print('\ttrain acc = {}\tap = {}\tf1 = {}\troc = {}'.format(
            round(accuracy_score(labels, (preds>0.5)*1)*100, 2), \
            round(average_precision_score(labels, preds)*100, 2),
            round(f1_score(labels, (preds>0.5)*1)*100, 2), 
            round(roc_auc_score(labels, preds)*100, 2)))
        print(train_0, train_1, (train_0+train_1)/2)
        
#         for data in val_loader:
#             data.to(device)
#             preds = model(data.x, data.edge_index, data.edge_type, data.batch).detach().numpy()
#             labels = data.y.detach().numpy()
        
#         comp = np.concatenate((((preds>0.5)*1).reshape(-1,1),labels.reshape(-1,1)),1)

#         val_0 = sum((comp[:,0]==0) * (comp[:,1]==0)*1) / sum((comp[:,1]==0)*1)
#         val_1 = sum((comp[:,0]==1) * (comp[:,1]==1)*1) / sum((comp[:,1]==1)*1)
        
#         print('\ttest acc = {}\tap = {}\tf1 = {}\troc = {}'.format(round(accuracy_score(labels, (preds>0.5)*1)*100, 2), \
#              round(average_precision_score(labels, preds)*100, 2), round(f1_score(labels, (preds>0.5)*1)*100, 2), round(roc_auc_score(labels, preds)*100, 2)))
        # print(val_0, val_1, (val_0+val_1)/2)
        
        for data in test_loader:
            data.to(device)
            preds = model(data.x, data.edge_index, data.edge_type, data.batch).detach().numpy()
            labels = data.y.detach().numpy()
        
        comp = np.concatenate((((preds>0.5)*1).reshape(-1,1),labels.reshape(-1,1)),1)

        test_0 = sum((comp[:,0]==0) * (comp[:,1]==0)*1) / sum((comp[:,1]==0)*1)
        test_1 = sum((comp[:,0]==1) * (comp[:,1]==1)*1) / sum((comp[:,1]==1)*1)
        
        print('\ttest acc = {}\tap = {}\tf1 = {}\troc = {}'.format(
            round(accuracy_score(labels, (preds>0.5)*1)*100, 2), \
             round(average_precision_score(labels, preds)*100, 2), 
             round(f1_score(labels, (preds>0.5)*1)*100, 2), 
             round(roc_auc_score(labels, preds)*100, 2)))
        print(test_0, test_1, (test_0+test_1)/2)
        
#         if max_acc <= (val_0+val_1)/2:
#             max_acc = (val_0+val_1)/2
#             max_epoch = epoch + 1
    # print(max_epoch)
def get_preds(probabilities, threshold=0.5):
        return [1 if prob > threshold else 0 for prob in probabilities]
def GVAE_pred(smi, enzyme, model_path='model/', device='cpu'): 
    if isinstance(smi, str): # smi is string
        smiles, nodes, edges, relations = preprocess_test([smi])
    elif isinstance(smi, list):
        smiles, nodes, edges, relations = preprocess_test(smi)
    # y = [0]*len(smiles)

    test_set = GDataset(nodes, edges, relations,y, range(len(smiles)))
    test_loader = DataLoader(test_set, batch_size=len(smiles), shuffle=False)

    model = torch.load(model_path+'GVAE'+ '_' + enzyme + '.pt')
    model.eval()
    for data in test_loader:
        data.to(device)
        preds = model(data.x, data.edge_index, data.edge_type, data.batch, 'fintune')
        # print(preds)
        # print(get_preds(preds)[0])
        
    return preds, get_preds(preds)[0]    

model = torch.load('model/GVAE_JAK1.pt')
print('finish loading')


smiles = ['O=C(NCCC(O)=O)C(C=C1)=CC=C1/N=N/C(C=C2C(O)=O)=CC=C2OCCOC3=CC=C(NC4=NC=C(C)C(NC5=CC=CC(S(NC(C)(C)C)(=O)=O)=C5)=N4)C=C3', 'OCCOC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(NC(C)(C)C)(=O)=O)=C3)=N2)C=C1']
smiles.extend(['C1CCC(C1)C(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3', 'CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N'])
smiles.extend(['CCS(=O)(=O)N1CC(C1)(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3', 'C1CC1C(=O)NC2=NN3C(=N2)C=CC=C3C4=CC=C(C=C4)CN5CCS(=O)(=O)CC5', 'CCC1CN(CC1C2=CN=C3N2C4=C(NC=C4)N=C3)C(=O)NCC(F)(F)F'])
smiles.extend(['OC(COC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(NC(C)(C)C)(=O)=O)=C3)=N2)C=C1)=O', 'O=C(NCCC(O)=O)C(C=C1)=CC=C1/N=N/C(C=C2C(O)=O)=CC=C2OCCOC3=CC=C(NC4=NC=C(C)C(NC5=CC=CC(S(N)(=O)=O)=C5)=N4)C=C3','OC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(NC(C)(C)C)(=O)=O)=C3)=N2)C=C1', 'OCCOC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(N)(=O)=O)=C3)=N2)C=C1'])
smiles.extend(['CC1=CN=C(N=C1NC2=CC(=CC=C2)S(=O)(=O)NC(C)(C)C)NC3=CC=C(C=C3)OCCN4CCCC4','C1CCN(C1)CCOC2=C3COCC=CCOCC4=CC(=CC=C4)C5=NC(=NC=C5)NC(=C3)C=C2'])
               
chemical_names = ['MMT3-72', 'MMT3-72-M2', 'Ruxolitinib', 'Tofacitinib', 'Baricitinib', 'Filgotinib', 'Upadacitinib']    
chemical_names.extend(['M3', 'M4', 'M5', 'M1', 'Fedratinib', 'Pacritinib'])

print(len(smiles))

def preprocess_test(smiles):
    nodes = []
    edges = []
    relations = []
    lens = []
    adjs = []
    ords = []
    for i in range(len(smiles)):
        node, adj, order = gen_smiles2graph(smiles[i])
        if node == 'error':
            print(i, smiles, 'error')
            continue
        lens.append(adj.shape[0])
        adjs.append(adj)  
        ords.append(order)
        node[:,2] += 1
        node[:,3] -= 1
        nodes.append(node)

    adjs = np.array(adjs)
    lens = np.array(lens)

    def file2array(path, delimiter=' '):     
        fp = open(path, 'r', encoding='utf-8')
        string = fp.read()              
        fp.close()
        row_list = string.splitlines()  
        data_list = [[float(i) for i in row.strip().split(',')] for row in row_list]
        return np.array(data_list)

    def adj2idx(adj):
        idx = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i,j] == 1:
                    idx.append([i,j])
        return np.array(idx)

    def order2relation(adj):
        idx = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i,j] != 0:
                    idx.extend([adj[i,j]])
        return np.array(idx)

    for i in range(lens.shape[0]):
        adj = adjs[i]
        order = ords[i]
        idx = adj2idx(adj)
        relation = order2relation(order)-1
        edges.append(idx)
        relations.append(relation)

    return smiles, nodes, edges, relations


smiles, nodes, edges, relations = preprocess_test(smiles)
y = [0]*len(smiles)

test_set = GDataset(nodes, edges, relations, y, range(len(smiles)))
test_loader = DataLoader(test_set, batch_size=len(smiles), shuffle=False)

# model = torch.load(rootpath+'GraphVAE.pt')
for data in test_loader:
    data.to(device)
    probs = model(data.x, data.edge_index, data.edge_type, data.batch, 'fintune')
    # print(preds)
preds = get_preds(probs)
print(preds)
# for smi, name in zip(smiles, chemical_names):
#     for enzyme in ['JAK1']:
#         pred = GVAE_pred(smi, enzyme, model_path='model/', device='cpu')
#         print('Name: ', name, 'enyzme: ', enzyme, ' ', pred)
# for name, pred in zip(chemical_names, preds):
#     print(name, pred)

# for name, smi in zip(chemical_names, smiles):
#     print(name)
#     a, b = GVAE_pred(smi, 'JAK1', model_path='model/', device='cpu')
#     print(a[0][0], ' and ', b)
# bar = 'CCS(=O)(=O)N1CC(C1)(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3'
# print(GVAE_pred(bar, 'JAK1', model_path='model/', device='cpu'))


GVAE_pred(smiles, 'JAK1')






# ###########
# from sklearn.metrics import classification_report
# # torch.save(GraphVAE, model_path + 'JAK2/' + filename)
# for enzyme in ['JAK1', 'JAK2', 'JAK3', 'TYK2']:
#     GraphVAE = torch.load('model/GVAE_' + enzyme + '.pt')

#     smiles, nodes, edges, relations, y = preprocess()
#     dataset = GDataset(nodes, edges, relations, y, range(0, len(smiles)))

#     train_num = int(0.8*len(smiles))
#     test_num = len(smiles) - train_num

#     _, data = torch.utils.data.random_split(dataset,[train_num, test_num], generator=torch.Generator().manual_seed(42))
#     dataloader = DataLoader(data, batch_size=test_num, shuffle=False)
#     for data in dataloader:
#         probas = GraphVAE(data.x, data.edge_index, data.edge_type, data.batch, 'finetune').detach().numpy()
#         labels = data.y.detach().numpy()
    
#     roc_values = []

#     def get_preds(threshold, probabilities):
#         return [1 if prob > threshold else 0 for prob in probabilities]
#     for thresh in np.linspace(0, 1, 2000):
#         preds = get_preds(thresh, probas)
#         tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
#         tpr = tp/(tp+fn)
#         fpr = fp/(fp+tn)
#         roc_values.append([tpr, fpr])
#     tpr_values, fpr_values = zip(*roc_values)

#     fig, ax = plt.subplots(figsize=(10,7))
#     ax.plot(fpr_values, tpr_values)
#     ax.plot(np.linspace(0, 1, 2000),
#             np.linspace(0, 1, 2000),
#             label='baseline',
#             linestyle='--')
#     plt.title('Receiver Operating Characteristic Curve', fontsize=18)
#     plt.ylabel('TPR', fontsize=16)
#     plt.xlabel('FPR', fontsize=16)
#     plt.legend(fontsize=12)

#     plt.savefig('ROC_AUC.png')

#     from function import save_tpr_fpr, load_tpr_fpr
#     save_tpr_fpr('GVAE', enzyme, tpr_values, fpr_values)


#     tpr, fpr = load_tpr_fpr('GVAE', enzyme)
#     fig, ax = plt.subplots(figsize=(10,7))
#     ax.plot(fpr, tpr)
#     ax.plot(np.linspace(0, 1, 2000),
#             np.linspace(0, 1, 2000),
#             label='baseline',
#             linestyle='--')
#     plt.title('Receiver Operating Characteristic Curve', fontsize=18)
#     plt.ylabel('TPR', fontsize=16)
#     plt.xlabel('FPR', fontsize=16)
#     plt.legend(fontsize=12)
#     plt.show()
