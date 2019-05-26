import torch
import torch.optim as optim
import torch.nn.functional as F
from metrics import accuracy
import time
from time import perf_counter
from utils import get_data_loaders


def train_mlp(model,
              train_features, 
              train_labels,
              val_features, 
              val_labels,
              epochs, 
              weight_decay,
              lr, 
              dropout,
              bs):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    train_loader, val_loader = get_data_loaders(train_features, 
                                                train_labels,
                                                val_features,
                                                val_labels,
                                                bs)
    t = perf_counter()
    max_acc_val = 0
    best_epoch = 0
    for epoch in range(epochs):
        for feats, labels in train_loader:
            model.train()
            optimizer.zero_grad()
            output = model(feats)
            loss_train = F.cross_entropy(output, labels)
            loss_train.backward()
            optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time


def train_regression(model,
                     train_features, 
                     train_labels,
                     val_features, 
                     val_labels,
                     epochs, 
                     weight_decay,
                     lr, 
                     dropout):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    max_acc_val = 0
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()

    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time


def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)


def train_gcn(model,
              adj,
              features, 
              labels,
              idx_train,
              idx_val, 
              epochs, 
              weight_decay,
              lr, 
              dropout):
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, 
                           weight_decay=weight_decay)

    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        acc_val = accuracy(output[idx_val], labels[idx_val])

    return model, acc_val, train_time


def test_gcn(model, adj, features, labels, idx_test):
    model.eval()
    output = model(features, adj)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test


def train_kgcn(model,
               adj,
               features, 
               labels,
               idx_train,
               idx_val, 
               epochs, 
               weight_decay,
               lr, 
               dropout):
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, 
                           weight_decay=weight_decay)

    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        acc_val = accuracy(output[idx_val], labels[idx_val])

    return model, acc_val, train_time


def test_kgcn(model, adj, features, labels, idx_test):
    model.eval()
    output = F.softmax(model(features, adj), dim=1)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test