from sklearn import datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np


def make_graph(X, n_neighbors=4, algo='ball_tree'):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algo).fit(X)
    A = nbrs.kneighbors_graph(X, mode='connectivity')
    return A


def make_donuts(n=4000, 
                noise=0.2, 
                factor=0.5, 
                test_size=0.92, 
                nneigh=5,
                mesh=False,
                mesh_step=0.02):
    X, y = datasets.make_circles(n_samples=n, noise=0.2, factor=0.5)
    adj = make_graph(X, nneigh)
    X = StandardScaler().fit_transform(X)
    sss = ShuffleSplit(n_splits=1, test_size=test_size)
    sss.get_n_splits(X, y)
    train_index, test_index = next(sss.split(X, y)) 
    mesh_X = None
    mesh_adj = None
    xx = None
    yy = None
    if mesh:
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                             np.arange(y_min, y_max, mesh_step))
        mesh_X = np.c_[xx.ravel(), yy.ravel()]
        mesh_adj = make_graph(mesh_X, nneigh)  # Might take a long time
    mesh_pack = (mesh_adj, mesh_X, xx, yy)
    return adj, X, y, train_index, test_index, test_index, mesh_pack


def make_bipartite(n=2000,
                   feature='noise',
                   feature_dim=20):
    pass 
