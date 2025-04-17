import numpy as np 
import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, least_squares
from scipy.linalg import cholesky, solve_triangular
from sklearn.linear_model import LinearRegression


seed = 10
np.random.seed(seed)

# 定义网络结构
G = nx.DiGraph()
G.add_edges_from([
    ('Source', 'A'),
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D'),
    ('C', 'D'),
    ('D', 'E')
])

sites = list(G.nodes())
n_sites = len(sites) # number of nodes

positions = {
    'Source': (0, 0),
    'A': (1, 0),
    'B': (2, 1),
    'C': (2, -1),
    'D': (3, 0),
    'E': (4, 0)
}

# 参数定义
time_steps = 5000  # 增加时间步数以获得更稳定的协方差估计
w = 2  # AR(w) 模型

# 真实参数（用于数据生成）
alpha_true = [0.7, 0.3]
sigma2_true = 1.0
phi_true = 2.0
tau2_true = 1.0
def compute_network_distance_matrix(G, sites, positions):
    n = len(sites)
    D = np.full((n, n), np.inf)
    for i, source in enumerate(sites):
        for j, target in enumerate(sites):
            if source == target:
                D[i, j] = 0
            elif nx.has_path(G, source=source, target=target):
                path = nx.shortest_path(G, source=source, target=target)
                distance = 0
                for k in range(len(path)-1):
                    coord1 = positions[path[k]]
                    coord2 = positions[path[k+1]]
                    distance += np.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)
                D[i, j] = distance
    return D
# 计算网络距离
def network_distance(G, source, target, positions):
    try:
        path = nx.shortest_path(G, source=source, target=target)
        distance = 0
        for i in range(len(path)-1):
            coord1 = positions[path[i]]
            coord2 = positions[path[i+1]]
            distance += np.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)
        return distance
    except nx.NetworkXNoPath:
        return np.inf  # 无路径时返回无穷大

# 构建空间协方差矩阵
def build_spatial_unbiased(D, sigma2, phi):
    Sigma_spatial = sigma2 * np.exp(-D / phi)
    Sigma_spatial[D == np.inf] = 0
    np.fill_diagonal(Sigma_spatial, sigma2)
    Sigma_spatial = np.triu(Sigma_spatial) + np.triu(Sigma_spatial, 1).T

    return Sigma_spatial
def build_spatial_gen(beta, phi, n, weights, dist_matrix):

    Sigma = np.zeros((n, n), dtype=float)
    for i in range(n):
        for k in range(n):
            if dist_matrix[i, k] > 1e3:
                Sigma[i, k] = 0
            else:
                dist_ik = dist_matrix[i, k]
                Sigma[i, k] = 0.5 * beta * np.sqrt(weights[i] / weights[k]) * np.exp(-dist_ik / phi)
    return Sigma

def build_spatial_covariance(args, beta, phi, n, weights, dist_matrix):
    if args.dataset == 'syn_tailup_gen':
        Sigma_spatial = build_spatial_gen(beta, phi, n, weights, dist_matrix)
    else: # args.dataset == 'syn_tailup':
        Sigma_spatial = build_spatial_unbiased(dist_matrix, beta, phi)
    return Sigma_spatial

def generate_tailup_data(D, sites, positions, time_steps, sigma2_true, phi_true):

    Sigma_spatial = build_spatial_covariance(D, sigma2_true, phi_true)
    print('Sigma_spatial', Sigma_spatial)

    # 初始化数据框架
    df = pd.DataFrame(index=sites, columns=range(time_steps))

    # 计算稳态方差（适用于 AR(w)）
    if 1 - sum([a**2 for a in alpha_true]) <= 0:
        raise ValueError("AR(w) is not stationary (1 - Σ(alpha_k^2) > 0)")

    steady_state_variance = tau2_true * sigma2_true / (1 - sum([a**2 for a in alpha_true]))

    # 从稳态分布中采样前 w 个时间点
    for t in range(w):
        for site in sites:
            df.at[site, t] = np.random.normal(0, np.sqrt(steady_state_variance))

    # 生成数据
    for t in range(w, time_steps):
        ar_matrix = np.zeros(n_sites)
        for i, site in enumerate(sites):
            for k in range(1, w+1):
                ar_matrix[i] += alpha_true[k-1] * df.at[site, t - k]
        epsilon = np.random.multivariate_normal(mean=np.zeros(n_sites), cov=Sigma_spatial )
        df.iloc[:, t] = ar_matrix + epsilon
    return Sigma_spatial, df

import configparser
def generate_config_file(config_path):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    config['data'] = {
        'num_nodes': n_sites,
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

    config['var_para'] = { 'w': w,
                           'D': np.array2string(D),
                           'alpha_true': np.array2string(np.array(alpha_true )),
                           'beta_true': sigma2_true,
                           'phi_true': phi_true,
                           'Sigma_spatial': np.array2string (Sigma_spatial )}

    if os.path.exists(config_path):
        print('Already Exists Config')
    else:
        with open( config_path , 'w') as configfile:
            config.write(configfile)
        print('Write Config')




# 估计自回归系数 alpha
def estimate_tailup_para(X, Y_response, Y , T, w , n_sites, alpha_true = None, sigma2_true = None, phi_true = None):

    X_flat = X.reshape(((T - w) * n_sites, w))
    Y_flat = Y_response.reshape((T - w) * n_sites)

    print("After reshaping: X", X_flat.shape, "Y", Y_flat.shape)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_flat, Y_flat)
    alpha_estimates = np.array(model.coef_)
    print("X", X_flat.shape, "Y_flat", Y_flat.shape, "alpha", alpha_estimates.shape)

    alpha_estimates = alpha_estimates[::-1]
    print(f"alpha Estimation: {alpha_estimates},\n Ground-truth: {alpha_true}")

    # Y = Y_flat.reshape( (T - w, n_sites) )
    print('Y', Y.shape)
    arr_str = np.array2string(
        Y[-10:,:],
        formatter={'float_kind': lambda x: f"{x:.8f}"}
    )
    print(arr_str)

    AR = np.zeros_like(Y)
    for k in range(1, w+1):
        AR[:, w:] += alpha_estimates[k-1] * Y[:, w-k:time_steps-k]

    residuals = (Y[:, w:] - AR[:, w:])#[:, -900:][ :, 0:300]

    print("residuals: ", residuals, 'SHAPE:', residuals.shape)

    residuals_centered = residuals - np.mean(residuals, axis=1, keepdims=True)
    C_empirical = (residuals_centered @ residuals_centered.T) / (residuals.shape[1]- w)
    print('C_empirical', C_empirical)

    initial_guess = [1.0, 1.0]
    result = least_squares(opt_objective, initial_guess, args=(D, C_empirical), bounds=(1e-5, np.inf))

    estimated_sigma2, estimated_phi = result.x


    print(f"sigma Estimation: {estimated_sigma2},\n Ground-truth: {sigma2_true}")
    print(f"phi Estimation: {estimated_phi},\n Ground-truth: {phi_true}")

    Sigma_spatial_estimated = build_spatial_covariance(D, estimated_sigma2, estimated_phi)


    return Sigma_spatial_estimated


def opt_objective(params, D, C_empirical):
    sigma2, phi = params
    C_model = build_spatial_covariance(D, sigma2, phi)
    # Mask for finite distances
    mask = D < np.inf
    error = (C_empirical[mask] - C_model[mask]).flatten()
    return error


if __name__ == "__main__":
    # time_steps*nodes
    D = compute_network_distance_matrix(G, sites, positions)
    print(D, D.dtype)
    Sigma_spatial, df = generate_tailup_data(D, sites, positions, time_steps, sigma2_true, phi_true)
    data = df.T.values.astype(float)
    print(data.shape)

    path = './data/syn_data/syn_tailup_{}.npz'.format(seed)
    config_path = './model/syn_tailup_{}.conf'.format(seed)

    if os.path.exists(path) and os.path.exists(config_path):
        print('Already Exists Dataset & Config')
    else:
        np.savez(path, array=data)
        generate_config_file(config_path)
        print('Created New Dataset & Config')

    Y = data.T
    X = []
    Y_response = []
    for t in range(w, time_steps):
        X.append(Y[:, t - w:t].flatten())
        Y_response.append(Y[:, t].flatten())
    print(Y[:, t - w:t].shape, Y[:, t - w:t].flatten().shape)
    print("top 10 data: ", Y[:10])

    T = time_steps
    X = np.array(X)  # shape: (T - w, n_sites * w)
    Y_response = np.array(Y_response)  # shape: (T - w, n_sites)
    print("Before reshaping: X", X.shape, "Y", Y_response.shape)

    Sigma_spatial_estimated = estimate_tailup_para(X, Y_response, Y , T, w , n_sites, alpha_true, sigma2_true, phi_true)
    print(Sigma_spatial_estimated)

'''
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
sns.heatmap(C_empirical, annot=True, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title("经验空间协方差矩阵 Σ_empirical")

plt.subplot(1, 3, 2)
sns.heatmap(Sigma_spatial_estimated, annot=True, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title("估计空间协方差矩阵 Σ_estimated")
plt.subplot(1, 3, 3)
sns.heatmap(Sigma_spatial, annot=True, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title("真实空间协方差矩阵 Σ_true")
plt.tight_layout()
plt.show()

'''




