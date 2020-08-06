import numpy as np
import scipy.sparse as sp
import torch


def fetch_normalization(type, **kwargs):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency, 
       'RwNorm': rw_normalized_adjacency,
       'InvLap': lambda adj: inv_normalized_laplacian(adj, **kwargs),
       'CombLap': comb_laplacian,
       'SymNormLap': sym_normalized_laplacian,
       'AbsRwNormAdj': abs_rw_normalized_adjacency
   }
   func = switcher.get(type, lambda x: x)
   return func

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def abs_rw_normalized_adjacency(adj): 
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1).flatten()
   d_inv[np.isinf(d_inv)] = 0.
   d_mat_inv = sp.diags(d_inv)
   return np.abs(d_mat_inv.dot(adj).tocoo())

def rw_normalized_adjacency(adj): 
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1).flatten()
   d_inv[np.isinf(d_inv)] = 0.
   d_mat_inv = sp.diags(d_inv)
   return d_mat_inv.dot(adj).tocoo()

def sym_normalized_laplacian(adj): 
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_mat = sp.diags(row_sum.flatten())
   L = d_mat - adj
   d_inv = np.power(row_sum, -1/2).flatten()
   d_inv[np.isinf(d_inv)] = 0.
   d_mat_inv = sp.diags(d_inv)
   return d_mat_inv.dot(L).dot(d_mat_inv).tocoo()

def inv_normalized_laplacian(adj, alpha=0.8):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_mat = sp.diags(row_sum.flatten())
   L = d_mat - adj
   d_inv = np.power(row_sum, -0.5).flatten()
   d_inv[np.isinf(d_inv)] = 0.
   d_mat_inv = sp.diags(d_inv)
   term = alpha * (sp.eye(adj.shape[0]) - d_mat_inv.dot(L).dot(d_mat_inv).tocoo())
   return term.dot(term)

def comb_laplacian(adj, scale=False):
   row_sum = np.array(adj.sum(1))
   d = sp.diags(row_sum.flatten())
   L = (d - adj).todense()
   if scale:
      max_eigval = np.max(np.linalg.eigvals(L))
      L = L / max_eigval
   L = sp.coo_matrix(L)
   return L

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
