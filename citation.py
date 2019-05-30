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
                  train_kgcn, test_kgcn, train_mlp

# Arguments
args = get_citation_args()

if args.tuned:
    if args.model == "SGC" or args.model == "KGCN":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# setting random seeds
set_seed(args.seed, args.cuda)

adj, features, labels, idx_train,\
idx_val, idx_test = load_citation(args.dataset,
                                  args.normalization,
                                  args.cuda,
                                  args.invlap_alpha,
                                  args.shuffle)

### NOISE TO FEATURES
if args.noise != "None":
    features = features.numpy()

if args.noise == "gaussian":
    features = gaussian(features,
                        mean=args.gaussian_opt[0],
                        std=args.gaussian_opt[1])
if args.noise == "gaussian_mimic":
    features = gaussian_mimic(features)
if args.noise == "add_gaussian":
    features = gaussian(features, 
                        mean=args.gaussian_opt[0],
                        std=args.gaussian_opt[1],
                        add=True)
if args.noise == "add_gaussian_mimic":
    features = gaussian_mimic(features, add=True)
if args.noise == "superimpose_gaussian":
    features = superimpose_gaussian(features, args.superimpose_k)
if args.noise == "superimpose_gaussian_class":
    labels = labels.numpy()
    features = superimpose_gaussian_class(features, labels)
    labels = torch.LongTensor(labels)
    if args.cuda:
        labels = labels.cuda()
if args.noise == "superimpose_gaussian_random":
    features = superimpose_gaussian_random(features, args.superimpose_k)
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


### STACKED FEATURES
#if args.stacked_feature:
#    #features = features.numpy()
#    features = stack_feat(features, adj, args.degree)
#    features = torch.FloatTensor(features).float()
#    if args.cuda:
#        features = features.cuda()
### END STACKED FEATURES

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
        model, acc_val, train_time = train_mlp(model, 
                                               features[idx_train],
                                               labels[idx_train], 
                                               features[idx_val], 
                                               labels[idx_val],
                                               args.epochs,
                                               args.weight_decay, 
                                               args.lr, 
                                               args.dropout,
                                               args.batch_size)
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