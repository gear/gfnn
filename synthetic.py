import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import sgc_precompute, set_seed, stack_feat, load_donuts
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_syn_args
from time import perf_counter
from noise import zero_idx, gaussian
from train import train_regression, test_regression,\
                  train_gcn, test_gcn,\
                  train_kgcn, test_kgcn

# Arguments
args = get_syn_args()

# setting random seeds
set_seed(args.seed, args.cuda)

adj, features, labels, idx_train,\
idx_val, idx_test, mesh_pack = load_donuts(args.gen_num_samples,
                                           args.gen_noise,
                                           args.gen_factor,
                                           args.gen_test_size,
                                           args.gen_num_neigh,
                                           args.normalization, 
                                           args.cuda,
                                           args.invlap_alpha,
                                           args.gen_mesh,
                                           args.gen_mesh_step) 

### NOISE TO FEATURES ONLY USE ZERO HERE
if args.noise != "None":
    features = features.numpy()

if args.noise == "gaussian":
    features = gaussian(features,
                        mean=args.gaussian_opt[0],
                        std=args.gaussian_opt[1])
if args.noise == "zero_test":
    idx_test = idx_test.numpy()
    features = zero_idx(features, idx_test)
    idx_test = torch.LongTensor(idx_test)
    if args.cuda:
        idx_test = idx_test.cuda()

if args.noise != "None":
    features = torch.FloatTensor(features).float()
    if args.cuda:
        features = features.cuda()
### END NOISE TO FEATURES

# Monkey patch for Stacked Logistic Regression
if args.model == "SLG":
    nfeat = features.size(1) * args.degree
else:
    nfeat = features.size(1)

model = get_model(model_opt=args.model,
                  nfeat=nfeat,
                  nclass=labels.max().item()+1,
                  nhid=args.hidden,
                  dropout=args.dropout,
                  cuda=args.cuda,
                  degree=args.degree)

if args.model == "SGC" or args.model == "SGCMLP": 
    features, precompute_time = sgc_precompute(features, adj, args.degree)
    print("{:.4f}s".format(precompute_time))
    model, acc_val, train_time = train_regression(model, 
                                                  features[idx_train],
                                                  labels[idx_train], 
                                                  features[idx_val], 
                                                  labels[idx_val],
                                                  args.epochs,
                                                  args.weight_decay, 
                                                  args.lr, 
                                                  args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])
    print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val,\
                                                                     acc_test))
    print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
if args.model == "SLG":
    features, precompute_time = stack_feat(features, adj, args.degree)
    features = torch.FloatTensor(features).float()
    if args.cuda:
        features = features.cuda()
    print("{:.4f}s".format(precompute_time))
    model, acc_val, train_time = train_regression(model, 
                                                  features[idx_train],
                                                  labels[idx_train], 
                                                  features[idx_val], 
                                                  labels[idx_val],
                                                  args.epochs,
                                                  args.weight_decay, 
                                                  args.lr, 
                                                  args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])
    print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val,\
                                                                     acc_test))
    print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))

if args.model == "GCN":
    model, acc_val, train_time = train_gcn(model, 
                                           adj,
                                           features,
                                           labels, 
                                           idx_train,
                                           idx_val,
                                           args.epochs,
                                           args.weight_decay,
                                           args.lr,
                                           args.dropout)
    acc_test = test_gcn(model, adj, features, labels, idx_test)
    print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val,\
                                                                     acc_test))
    precompute_time = 0
    print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))

if args.model == "KGCN":
    model, acc_val, train_time = train_kgcn(model, 
                                            adj,
                                            features,
                                            labels, 
                                            idx_train,
                                            idx_val,
                                            args.epochs,
                                            args.weight_decay,
                                            args.lr,
                                            args.dropout)
    acc_test = test_kgcn(model, adj, features, labels, idx_test)
    precompute_time = 0
    print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val,\
                                                                     acc_test))
    print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
