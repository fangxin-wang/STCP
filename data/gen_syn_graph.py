import sys
import os
print(os.path.curdir)
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import numpy as np
import pandas as pd
import torch
from lib.TrainInits import init_seed
import argparse
import configparser
import scipy.sparse as sp
# from data.gen_syn_data import generate_config_file

from scipy.stats import multivariate_normal

#parser
parser = argparse.ArgumentParser( )
parser.add_argument('--syn_seed', default= 0, type=int)
parser.add_argument("--node_num", default = 5, type = int)
parser.add_argument("--T_num", default = 20000, type = int)
parser.add_argument("--L", default = 2, type = int)
parser.add_argument("--Q", default = 12, type = int)
parser.add_argument("--sample_size", default = 1000, type = int, help = "Sample Size For Simulating GT Distribution")


args = parser.parse_args()
seed = args.syn_seed
init_seed(seed)

print("Initialized")


def generate_config_file(N, adj_matrix, correlation_matrix, config_path):
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    if correlation_matrix is None:
        raise ValueError("correlation_matrix is None")

    config['data'] = {
        'num_nodes': N,
        'lag': 12,
        'horizon': 1,
        'val_ratio' : '0.2',
        'test_ratio' : '0.2',
    'tod' : False, 'normalizer' : 'std', 'column_wise' : False, 'default_graph' : True}

    config['model'] = {
        'input_dim': '1',
        'output_dim': '1',
        'embed_dim': '10',
        'rnn_units': '64',
        'num_layers': '2',
        'cheb_order': '2',
        'p1': '0.1'
    }

    # [train] section
    config['train'] = {
        'loss_func': 'mae',
        'seed': '10',
        'batch_size': '64',
        'epochs': '100',
        'lr_init': '0.003',
        'lr_decay': 'False',
        'lr_decay_rate': '0.3',
        'lr_decay_step': '5,20,40,70',
        'early_stop': 'True',
        'early_stop_patience': '15',
        'grad_norm': 'False',
        'max_grad_norm': '5',
        'real_value': 'True'
    }

    # [test] section
    config['test'] = {
        'mae_thresh': 'None',
        'mape_thresh': '0.001'
    }

    # [log] section
    config['log'] = {
        'log_step': '20',
        'plot': 'False'
    }

    config['var_para'] = {
        'adj_m': np.array2string(adj_matrix),
        'cor_m': np.array2string(correlation_matrix.numpy()),
        'cor_hop': L,
        'cor_t': Q,
        'Theta': np.array2string( Theta.numpy() ),
        'noise_mu':np.array2string( mu.numpy() ),
        'noise_sigma': np.array2string(sigma),
    }

    if os.path.exists(config_path):
        print('Already Exists Config')
    else:
        with open( config_path , 'w') as configfile:
            config.write(configfile)
        print('Write Config')


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^-0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def generate_A_S(N):
    # Generate a random binary adjacency matrix (0 or 1 values)
    A = np.random.binomial(1,0.5, size=(N, N))  # Random binary matrix with values 0 or 1
    A = np.triu(A, 1)  # Make it upper triangular to avoid self-loops and symmetry
    A = A + A.T  # Make it symmetric (undirected graph)
    print("A:")
    print(A)
    A_sparse = sp.coo_array(A)
    S = preprocess_adj(A_sparse).todense()

    print("S:")
    print(S,S.sum(1))
    return A, S


def GPVAR_1_step(x, t, N_s, N, L, Q, S, Theta):
    """
    Generate GT observations at time step t.

    :param x: Ground-truth generated observation
    :param t: Time step
    :param N_s: Number of samples at time t
    :param N: Number of nodes
    :param L: See GPVAR
    :param Q: See GPVAR
    :param S: See GPVAR
    :param Theta: See GPVAR
    :return: as N_s*N shape
    """
    if t<Q:
        raise ValueError("Time step must be greater than Correlation Number")

    Res = torch.zeros(N_s, N)
    eta = lambda size: torch.randn(size)
    sum_term = torch.zeros(N)

    for s in range(N_s):
        for l in range(L + 1):
            for q in range(1, Q):
                shifted_x = torch.matrix_power(S, l) @ x[:, t - q - 1]
                sum_term += Theta[l, q] * shifted_x
        Res[s, :] = torch.tanh(sum_term) + 0.1 *  eta(N)

    if t%10 == 0:
        print("Finish generating {} samples at {} step".format(N_s,t) )
    return Res

def generate_symmetric_positive_semidefinite_matrix(size):
    # 生成一个随机矩阵
    random_matrix = np.random.rand(size, size)
    # 通过矩阵乘法得到对称半正定矩阵
    symmetric_positive_semidefinite_matrix = random_matrix.T @ random_matrix
    return symmetric_positive_semidefinite_matrix

def GPVAR(N, L, Q, T, S, Theta):
    """
    Generating GP-VAR Model.
    https://arxiv.org/pdf/2204.11135

    :param N: Number of nodes
    :param L: Number of correlated hops
    :param Q: Number of correlated observations in the past
    :param T: Number of time steps
    :param S: Graph shift operator
    :param Theta: Parameters of correlation scale: (L+1)*Q shape
    :return: Return as N*T shape
    """
    x = torch.zeros(N, T)  # Node signals over time

    ## Multivariate Gaussian Distribution
    mu = torch.randn(N)
    sigma = generate_symmetric_positive_semidefinite_matrix(N)
    dist = multivariate_normal(mean=mu, cov=sigma)

    eta = lambda size: torch.randn(size)  # White noise from standard Gaussian distribution

    # Generate the synthetic data
    for t in range(Q, T):
        sum_term = torch.zeros(N)
        for l in range(L+1):
            for q in range(Q):
                shifted_x = torch.matrix_power(S, l) @ x[:, t - q - 1]
                sum_term += Theta[l, q] * shifted_x
        # x[:, t] = torch.tanh(sum_term) + 0.1 * eta(N)
        x[:, t] = torch.tanh(sum_term) + 0.1 * dist.rvs(size=1)
        # print('sum: ', torch.tanh(sum_term), 'noise:', 0.1 * dist.rvs(size=1), 'x[:, t]: ', x[:, t])

    print("Generated data shape:", x.shape)
    print("Sample data for first node:", x[0, :100])
    return x, mu, sigma


def save_files(N, A, S, x):
    path = './data/syn_data/syn_gpvar_{}.npz'.format(seed)
    config_path = './model/syn_gpvar_{}.conf'.format(seed)
    # sample_path = './data/syn_data/syn_gpvar_sample_{}.npz'.format(seed)

    generate_config_file(N, A, S, config_path)

    if os.path.exists(path): #and os.path.exists(sample_path):
        print('Already Exists Dataset')
    else:
        np.savez(path, array= x)
        # np.savez(sample_path, array=sample_res)
        print('Created New Dataset')

N, L, Q, T = args.node_num, args.L, args.Q, args.T_num
print( N, L, Q , T)
# def generate_STGraph
Theta = torch.randint(-10, 10, (L+1, Q))
print('Theta: ',Theta)
A, S = generate_A_S( N )

S = torch.from_numpy(S).float()
x, mu, sigma = GPVAR( N,  L, Q, T, S, Theta)
x = x.T
print(x.shape) #  [20000, 5] time_steps*nodes

### test

Sample_size = args.sample_size

def generate_all_time_samples(x, Sample_size, N, L, Q, S, Theta):

    # N_s*N  shape
    Sampled_res = []
    for t in range(Q, T):
        Sampled_res_t = GPVAR_1_step(x, t, Sample_size, N, L, Q, S, Theta)
        Sampled_res.append(Sampled_res_t)

    # # Check
    print("Orginal value at time {} is {}".format( t, x[:,t])  )
    print("First 3 Samples:", Sampled_res_t[:3,:])

    return np.array(Sampled_res)

save_files(N, A, S, x)

# Sampled_res = generate_all_time_samples(x, Sample_size, N, L, Q, S, Theta)
# print(Sampled_res.shape)
#
# ### FINAL SAVE
# save_files(N, A, x, Sampled_res)
