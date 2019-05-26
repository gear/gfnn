import time 
import argparse 
import numpy as np 
import sklearn as sk
from utils import load_citation
from normalization import fetch_normalization
from transformation import fetch_transformation
from train import train_regression, test_regression
from models import get_model
import torch
from args import get_feat_args


datasets = [
    "cora",
    "citeseer",
    "pubmed"
]

preps = [
    '',
    'GFT'
]

normalization_choices = [
    '',
    'AugNormAdj',
    'LeftNorm',
    'InvLap',
    'CombLap',
    'SymNormAdj',
    'SymNormLap'
]


args = get_feat_args()

adj, features, labels, idx_train,\
    idx_val, idx_test = load_citation(args.dataset, 
                                      normalization=args.normalization, 
                                      cuda=False) 


model = get_model(model_opt=args.model,
                  nfeat=features.size(1),
                  nclass=labels.max().item()+1,
                  nhid=args.hidden,
                  dropout=args.dropout,
                  cuda=False)


# TODO: Calculate time here
features = features.numpy()
transformer = fetch_transformation(args.preprocess)
forward, invert, evals = transformer(adj.to_dense())
features = invert(forward(features, 
                          i=args.first_component,\
                          k=args.num_component),
                  i=args.first_component,
                  k=args.num_component)
features = torch.FloatTensor(features).float()

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