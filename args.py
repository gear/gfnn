import argparse
import torch


noise_choices = [
    'None',
    'gaussian',
    'gaussian_mimic',
    'add_gaussian',    
    'add_gaussian_mimic',
    'superimpose_gaussian',
    'superimpose_gaussian_class',
    'superimpose_gaussian_random',
    'zero_test'
]

normalization_choices = [
    '',
    'AugNormAdj',
    'LeftNorm',
    'InvLap',
    'CombLap',
    'SymNormLap'
]

# "All neural network models should go to work at Victoria's Secret
#  so the VS models would be here, with us."
sexy_models = [
    'SGC',
    'GCN',
    'KGCN',
    'SLG',
    'gfnn'
]

preps = [
    '',
    'GFT'
]

def get_feat_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--model', type=str, default="SGC",
                        choices=sexy_models,
                        help='model to use.')
    parser.add_argument('--feature', type=str, default="mul",
                        choices=['mul', 'cat', 'adj'],
                        help='feature-type')
    parser.add_argument('--normalization', type=str, default='LeftNorm',
                       choices=normalization_choices,
                       help='Normalization method for the adjacency matrix.')
    parser.add_argument('--invlap_alpha', type=float, default=0.5,
                        help='alpha parameter for InvLap norm.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    parser.add_argument('--noise', type=str, default='None', 
                        choices=noise_choices, help='noise settings')
    parser.add_argument('--preprocess', type=str, choices=preps, default="GFT")
    parser.add_argument('--num_component', type=int, default=1)
    parser.add_argument('--first_component', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', action='store_true')

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def get_syn_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--data', type=str, default="bicircle",
                        help='Data shape to generate.')
    parser.add_argument('--model', type=str, default="SGC",
                        choices=sexy_models,
                        help='model to use.')
    parser.add_argument('--feature', type=str, default="mul",
                        choices=['mul', 'cat', 'adj'],
                        help='feature-type')
    parser.add_argument('--normalization', type=str, default='LeftNorm',
                       choices=normalization_choices,
                       help='Normalization method for the adjacency matrix.')
    parser.add_argument('--invlap_alpha', type=float, default=0.5,
                        help='alpha parameter for InvLap norm.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    parser.add_argument('--per', type=int, default=-1,
                        help='Number of each nodes so as to balance.')
    parser.add_argument('--noise', type=str, default='None', 
                        choices=noise_choices, help='noise settings')
    parser.add_argument('--gaussian_opt', type=float, nargs=2, 
                        default=[0.0, 1.0], help="mean and var for gaussian")
    parser.add_argument('--gen_num_samples', type=int, default=4000,
                        help='Number of synthetic sample to generate.')
    parser.add_argument('--gen_noise', type=float, default=0.2,
                        help='Amount of noise added to generated samples')
    parser.add_argument('--gen_factor', type=float, default=0.5, 
                        help='Scaling factor for circle data generation.')
    parser.add_argument('--gen_test_size', type=float, default=0.98,
                        help='Amount of data to be used as test.')
    parser.add_argument('--gen_num_neigh', type=int, default=5,
                        help='Number of neighbors to build the graph.')
    parser.add_argument('--gen_mesh', action='store_true',
                        help='Generate a mesh for coutour plots')
    parser.add_argument('--gen_mesh_step', type=float, default=0.02,
                        help='Number of steps for the mesh') 
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', action='store_true')

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--model', type=str, default="SGC",
                        choices=sexy_models,
                        help='model to use.')
    parser.add_argument('--feature', type=str, default="mul",
                        choices=['mul', 'cat', 'adj'],
                        help='feature-type')
    parser.add_argument('--normalization', type=str, default='LeftNorm',
                       choices=normalization_choices,
                       help='Normalization method for the adjacency matrix.')
    parser.add_argument('--invlap_alpha', type=float, default=0.5,
                        help='alpha parameter for InvLap norm.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    parser.add_argument('--per', type=int, default=-1,
                        help='Number of each nodes so as to balance.')
    parser.add_argument('--experiment', type=str, default="base-experiment",
                        help='feature-type')
    parser.add_argument('--tuned', action='store_true', 
                        help='use tuned hyperparams')
    parser.add_argument('--noise', type=str, default='None', 
                        choices=noise_choices, help='noise settings')
    parser.add_argument('--gaussian_opt', type=float, nargs=2, 
                        default=[0.0, 1.0], help="mean and var for gaussian")
    parser.add_argument('--superimpose_k', type=float, default=1.5, 
                        help="extends feature vecs by k times")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', action='store_true')

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
