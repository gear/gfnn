import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed, stack_feat
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter
from noise import gaussian, gaussian_mimic,\
                  superimpose_gaussian, superimpose_gaussian_class,\
                  superimpose_gaussian_random, zero_idx
from train import train_regression, test_regression,\
                  train_gcn, test_gcn,\
                  train_kgcn, test_kgcn, train_mlp, train_gfnn

# Arguments
args = get_args()

# setting random seeds
set_seed(args.seed, args.cuda)

adj, features, labels, idx_train,\
idx_val, idx_test = load_data(args.dataset,
                              args.normalization.split("_"),
                              args.cuda,
                              args.invlap_alpha,
                              args.shuffle)

# Monkey patch for Stacked Logistic Regression
if args.model == "SLG":
    nfeat = features.size(1) * (args.degree+1)
else:
    nfeat = features.size(1)

model = get_model(model_opt=args.model,
                  nfeat=nfeat,
                  nclass=labels.max().item()+1,
                  nhid=args.hidden,
                  dropout=args.dropout,
                  cuda=args.cuda,
                  degree=args.degree)

if args.model == "SGC" or args.model == "gfnn":  
    features, precompute_time = sgc_precompute(features, adj, args.degree)
    print("{:.4f}s".format(precompute_time))
    if args.model == "gfnn":
        model, acc_val, train_time = train_gfnn(model, 
                                                features[idx_train],
                                                labels[idx_train], 
                                                features[idx_val], 
                                                labels[idx_val],
                                                epochs=args.epochs, 
                                                weight_decay=args.weight_decay, 
                                                lr=args.lr, 
                                                bs=args.batch_size,
                                                patience=800,
                                                verbose=True)
    else:
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
