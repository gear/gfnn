import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """
    Simple two layers GCN
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class KGCN(Module):
    """
    A bit more complex GNN to deal with non-convex feature space.
    """
    def __init__(self, nhidden, nfeat, nclass, degree):
        super(KGCN, self).__init__()
        self.Wx = GraphConvolution(nfeat, nhidden)
        self.W = nn.Linear(nhidden, nclass)
        self.d = degree

    def forward(self, x, adj):
        h = F.relu(self.Wx(x, adj))
        for i in range(self.d):
            h = torch.spmm(adj, h)
        return self.W(h)


class SGC(Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)


class MLP(Module):
    """
    A Simple two layers MLP to make SGC a bit better.
    """
    def __init__(self, nfeat, nhid, nclass, dp=0.2):
        super(MLP, self).__init__()
        self.W1 = nn.Linear(nfeat, nhid)
        self.W2 = nn.Linear(nhid, nclass)
        self.dp = dp

    def forward(self, x):
        x = F.relu(self.W1(x))
        x = nn.Dropout(p=self.dp)(x)
        return self.W2(x)
        

class SLG(Module):
    """
    Stacked feature with logreg.
    TODO: It doesn't make sense to dropout the final layer, need to add one more
    layer to perform dropout in between
    """
    def __init__(self, nfeat, nclass, dp=0.2):
        super(SLG, self).__init__()
        self.dp = dp
        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return nn.Dropout(p=self.dp)(self.W(x))


def get_model(model_opt, nfeat, nclass, 
              nhid=10, dropout=0, cuda=True, degree=2):
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)  
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nclass=nclass)
    elif model_opt == "KGCN":
        model = KGCN(nhidden=nhid,
                     nfeat=nfeat,
                     nclass=nclass,
                     degree=degree)
    elif model_opt == "SLG":
        model = SLG(nfeat=nfeat,
                    nclass=nclass,
                    dp=dropout)
    elif model_opt == "gfnn":
        model = MLP(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dp=dropout)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model