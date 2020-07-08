import random
import numpy as np
import scipy.sparse as sp
import torch
import sys
import os
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from synthetic_data import make_donuts
from time import perf_counter
from torch.utils import data
from sklearn.model_selection import train_test_split

dataf = os.path.expanduser("{}/data/".format(os.path.dirname(__file__)))


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization, extra=None):
    adj_normalizer = fetch_normalization(normalization, extra)
    adj = adj_normalizer(adj)
    #row_sum = 1 / (np.sqrt(np.array(adj.sum(1))))
    #row_sum = np.array(adj.sum(1))
    #features = row_sum
    #features = features.todense()
    #features = np.concatenate([features, row_sum], axis=1) 
    #features = sp.lil_matrix(features)
    if normalization != "":
        features = row_normalize(features)
    return adj, features


def preprocess_synthetic(adj, features, normalization, extra=None):
    adj_normalizer = fetch_normalization(normalization, extra)
    adj = adj_normalizer(adj)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(dataset_str="cora",
              normalization=[],
              feat_normalize=True,
              cuda=False,
              split="default",
              **kwargs):
    """
    Load pickle packed datasets.
    """
    with open(dataf+dataset_str+".graph", "rb") as f:
        graph = pkl.load(f)
    with open(dataf+dataset_str+".X", "rb") as f:
        X = pkl.load(f)
    with open(dataf+dataset_str+".y", "rb") as f:
        y = pkl.load(f)
    if split != "default":
        tr_size, va_size, te_size = [float(i) for i in split.split("_")]
        idx_train, idx_val, idx_test = \
            train_val_test_split(np.arange(len(y)), train_size=tr_size,
                                 val_size=va_size, test_size=te_size,
                                 stratify=y, random_state=None) 
    else:
        with open(dataf+dataset_str+".split", "rb") as f:
            split = pkl.load(f)
            idx_train = split['train']
            idx_test = split['test']
            idx_val = split['valid']

    normed_adj = []
    if len(normalization) > 0:
        adj = nx.adj_matrix(graph)
        for n in normalization:
            nf = fetch_normalization(n, **kwargs)
            normed_adj.append(nf(adj))

    if feat_normalize:
        X = row_normalize(X)

    X = torch.FloatTensor(X).float()
    y = torch.LongTensor(y)
    normed_adj = [sparse_mx_to_torch_sparse_tensor(adj).float() \
                  for adj in normed_adj]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        X = X.cuda()
        normed_adj = [adj.cuda() for adj in normed_adj]
        y = y.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return graph, normed_adj, X, y, idx_train, idx_val, idx_test


def load_citation(dataset_str="cora", 
                  normalization="AugNormAdj", 
                  cuda=True, 
                  extra=None,
                  shuffle=False,
                  train_test_val=False):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(dataf+"gcn/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(dataf+"gcn/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    if shuffle:
        n = len(idx_test) + len(idx_train) + len(idx_val)
        ids = [*range(n)]
        random.shuffle(ids)
        nidx_test = ids[:len(idx_test)]
        nidx_train = ids[len(idx_test):len(idx_test)+len(idx_train)]
        nidx_val = ids[len(idx_test)+len(idx_train):]

        idx_test = nidx_test
        idx_train = nidx_train
        idx_val = nidx_val
        

    adj, features = preprocess_citation(adj, features, normalization, extra)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if train_test_val:
        idx_train, idx_val, idx_test = \
                       train_val_test_split_tabular(np.arange(labels.size(0)),
                                                    train_size=0.1, 
                                                    val_size=0.1, 
                                                    test_size=0.8,
                                                    stratify=labels.numpy().ravel(), 
                                                    random_state=None)        

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def train_val_test_split(*arrays, 
                         train_size=0.5, 
                         val_size=0.3, 
                         test_size=0.2, 
                         stratify=None, 
                         random_state=None):

    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result


def load_donuts(n=4000, 
                noise=0.2, factor=0.5, test_size=0.92, nneigh=5,
                normalization='AugNormAdj', cuda=False, extra=None,
                mesh=False, mesh_step=0.02):
    adj, features, labels, idx_train, idx_val, idx_test,\
                 mesh_pack = make_donuts(n=n, 
                                         noise=noise, 
                                         factor=factor, 
                                         test_size=test_size, 
                                         nneigh=nneigh, 
                                         mesh=mesh, 
                                         mesh_step=mesh_step)
    mesh_adj, mesh_X, xx, yy = mesh_pack
    adj, features = preprocess_synthetic(adj, features, normalization, extra) 
    # porting to pytorch
    features = torch.FloatTensor(features).float()
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    if mesh:
        mesh_adj, mesh_X = preprocess_synthetic(mesh_adj, 
                                                mesh_X, 
                                                normalization, 
                                                extra)
        mesh_adj = sparse_mx_to_torch_sparse_tensor(mesh_adj).float()
        mesh_X = torch.FloatTensor(mesh_X).float()

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        if mesh:
            mesh_adj = mesh_adj.cuda()
            mesh_X = mesh_X.cuda()
    
    mesh_pack = (mesh_adj, mesh_X, xx, yy)

    return adj, features, labels, idx_train, idx_val, idx_test, mesh_pack

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def stack_feat(features, adj, degree):
    t = perf_counter()
    features_list = []
    #features_list = []
    for i in range(degree):
        features = torch.spmm(adj, features)
        features_list.append(features.numpy())
    precompute_time = perf_counter()-t
    features = np.concatenate(features_list, axis=1)
    return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(normalization="AugNormAdj", data_path=dataf+"reddit/",cuda=False, extra=None):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(data_path)
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization, extra)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index


class FeaturesData(data.Dataset):
    def __init__(self, X, y):
        self.labels = y
        self.features = X

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # Select sample
        X = self.features[index]
        y = self.labels[index]
        return X, y


class LowHighFreqData(torch.utils.data.Dataset):
    def __init__(self, X_low, X_high, y):
        self.labels = y
        self.features_low = X_low
        self.features_high = X_high
        self.length = y.size(-1)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        Xl = self.features_low[index]
        Xh = self.features_high[index]
        y = self.labels[index]
        return Xl, Xh, y


def low_high_data_loader(X_train_low, X_train_high, y_train, 
                         X_val_low, X_val_high, y_val, batch_size=32):
    train_set = LowHighFreqData(X_train_low, X_train_high, y_train)
    val_set = LowHighFreqData(X_val_low, X_val_high, y_val)
    trainLoader = torch.utils.data.DataLoader(dataset=train_set, 
                                              batch_size=batch_size, 
                                              shuffle=True)
    valLoader = torch.utils.data.DataLoader(dataset=val_set, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    return trainLoader, valLoader


def get_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    train_set = FeaturesData(X_train, y_train)
    val_set = FeaturesData(X_val, y_val)
    trainLoader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valLoader = data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
    return trainLoader, valLoader
