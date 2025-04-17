
import numpy as np

import torch
import scipy.special 
import math
import alphashape
import matplotlib.pyplot as plt

import networkx as nx
import pickle
import torch.nn.functional as F

from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize, least_squares
# from data.gen_syn_graph_tailup import opt_objective, build_spatial_covariance


def read_G(syn_seed):
    path = "./data/syn_data/G_seed_{}.gpickle".format(syn_seed)
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G

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
            if dist_matrix[i, k] > 1e10 and dist_matrix[k, i] > 1e10:
                Sigma[i, k],  Sigma[k, i] = 0, 0
            elif dist_matrix[i, k] <= 1e10:
                Sigma[i, k] = 0.5 * beta ** 2 * np.sqrt(weights[i] / weights[k]) * np.exp(-dist_matrix[i, k] / phi)
            else:
                Sigma[i, k] = 0.5 * beta ** 2 * np.sqrt(weights[k] / weights[i]) * np.exp(-dist_matrix[k, i] / phi)
    return Sigma

def build_spatial_covariance(dataset, beta, phi, n, weights, dist_matrix):
    if  dataset == 'syn_tailup':
        Sigma_spatial = build_spatial_unbiased(dist_matrix, beta, phi)
    else:
        Sigma_spatial = build_spatial_gen(beta, phi, n, weights, dist_matrix)
        # A_symmetric = A + A.T - np.diag(np.diag(A))
        # Sigma_spatial = Sigma_spatial + Sigma_spatial.T - np.diag(np.diag(Sigma_spatial))
    return Sigma_spatial

def opt_objective(params, C_empirical, dist_matrix, n, weights, dataset):
    beta, phi = params
    C_model = build_spatial_covariance(dataset, beta, phi, n, weights, dist_matrix)
    # Mask for finite distances
    mask = dist_matrix < np.inf
    # print(C_empirical.shape, C_model.shape, mask.shape)
    error = (C_empirical[mask] - C_model[mask]).flatten()
    abs_error =  np.mean(np.abs(error) )
    # print('C_empirical', C_empirical, 'C_model', C_model )
    # print('paras',beta, phi, 'abs_error', abs_error)
    return abs_error

def opt_objective_weights(params, C_empirical, dist_matrix, n, dataset):
    beta, phi = params[0], params[1]
    weights = params[1:]
    C_model = build_spatial_covariance(dataset, beta, phi, n, weights, dist_matrix)
    # Mask for finite distances
    mask = dist_matrix < np.inf
    # print(C_empirical.shape, C_model.shape, mask.shape)
    error = (C_empirical[mask] - C_model[mask]).flatten()
    abs_error =  np.mean(np.abs(error) )
    print('abs_error', abs_error)
    return abs_error

def ellipsoid_volume(covariance_matrix, r):
    # compute volume of the ellipsoid along for radius r
    eigenvalues = np.linalg.eigvals(covariance_matrix)
    eigenvalues = np.real(np.sort(eigenvalues)[::-1])
    eps = 1e-6
    num_r = np.sum(eigenvalues > eps)
    #det_sigma = np.prod( eigenvalues[:num_r])
    det_sigma = np.prod( np.sqrt(eigenvalues[:num_r]) )
    constant_cd = np.pi**(num_r/2) / np.math.gamma(num_r/2 + 1)  # Volume constant for d-dimensional sphere
    volume = constant_cd * r**num_r * det_sigma #np.sqrt(det_sigma)
    # print( constant_cd, r, num_r, eigenvalues[:num_r], det_sigma)
    return volume


def sphere_volume(d, r):
    # Gamma function can be accessed through math.gamma
    # ** (1 / d)
    return (math.pi ** (1 / 2) * r) / math.gamma(d / 2 + 1) ** (1 / d)
    #return (math.pi ** (d / 2) * r ** d) / math.gamma(d / 2 + 1)
#
# def PCP(model,data_loader,alpha,args):
#     y_true=[]
#     y_pred=[]
#     x=[]
#     #model.to(args.device)
#     with torch.no_grad():
#             for batch_idx, (data, target) in enumerate(data_loader):
#                 data = data[..., :args.input_dim]
#                 label = target[..., :args.output_dim]
#                 temp=[]
#                 for _ in range(args.K):
#                     output = model.predict(data, target, teacher_forcing_ratio=0)
#                     temp.append(output)
#                 temp1=torch.cat(temp,dim=1)
#                 y_pred.append(temp1)
#
#                 y_true.append(label)
#                 x.append(data)
#     x=torch.cat(x,dim=0) #shape[3988,12,5,1]
#     y_pred=torch.cat(y_pred,dim=0) #shape[3988,K,5,1]
#     y_true=torch.cat(y_true,dim=0)#shape[3988,1,5,1]
#
#     num_node=y_true.shape[2]
#
#     T = len(y_true)
#
#     alphatlist=[]
#     adaptErrSeq =torch.zeros((T - args.tinit))
#     alphat = alpha
#     eff=0
#     for t in range(args.tinit, T):
#
#         predcal, YCal =y_pred[t-args.tinit:t],y_true[t - args.tinit:t]  # calibration set y_pred [tinit，k，5，1] ytrue: [tinit,1，5，1]
#         # Compute nonconformity scores for the calibration set
#         distances = torch.norm(predcal.squeeze(-1) - YCal.squeeze(-1), dim=-1)  # ||y_pred - y_true||_2
#
#         nonconformity_scores, _ = distances.min(dim=1)
#         # Convert nonconformity scores to a tensor
#         nonconformity_scores = torch.tensor(nonconformity_scores)
#         alphat = np.clip(alphat, 0, 1)
#         a = torch.tensor(1 - alphat, device = nonconformity_scores.device )
#         b = torch.ceil((args.tinit + 1) * a) / args.tinit
#         b = torch.clip(b, 0, 1).to(nonconformity_scores.dtype)
#         # print("the shape of b is {}".format(b.shape))
#         q_alpha = torch.quantile(nonconformity_scores, b, interpolation='higher')
#
#         # Get predictions for the t-th test sample
#         y_pred_t = y_pred[t]  # Shape: (K, 5, 1)
#         y_true_t = y_true[t]  # Shape: (5, 1)
#
#         # Compute conformity for the t-th test sample
#         distances = torch.norm(y_pred_t.squeeze(-1) - y_true_t.squeeze(-1), dim=-1)  # ||y_pred - y_true||_2
#
#         dist=torch.min(distances)
#
#         adaptErrSeq[t - args.tinit] = (dist > q_alpha).float()
#
#         alphat += args.gamma * (alpha - adaptErrSeq[t - args.tinit])
#         # Store the adaptive alpha at time t
#         alphatlist.append(alphat)
#         eff += args.K * sphere_volume(num_node,q_alpha)
#     picp = 1 - torch.sum(adaptErrSeq) / len(adaptErrSeq)
#
#     print(f"average of picp is {picp} and inefficiency is {eff}")
#     return picp,eff
def compare_gt_error(C_empirical, args):
    path_gt_cov = './data/syn_data/gt_cov_{}.npz'.format(args.syn_seed)
    A = read_npz(path_gt_cov)
    gt_cov = A + A.T - np.diag(np.diag(A))
    # print(gt_cov)
    # print('gt_cov_deg', np.linalg.det(gt_cov))
    C_empirical = np.array(C_empirical)
    return np.mean( np.abs( C_empirical -  gt_cov ))

def get_cov_spatial(x, t, tinit, YCal, args):
    num_nodes = args.num_nodes
    Y_flat = YCal.squeeze()
    Y_t = np.array(Y_flat.cpu().numpy()).T
    t_num = Y_t.shape[1]

    if args.w > 0:

        X = x[t - tinit:t, -args.w:, :]
        X = X.squeeze(-1)
        X = X.permute(0, 2, 1)
        X_flat = X
        # X_flat = X.reshape(X.size(0), -1)

        # print(X_flat[-3:,-1,:], Y_flat[-3:,-1])

        # X_flat, Y_flat = X_flat[args.w : ], Y_flat[args.w : ]
        # print("X_flat", X_flat.shape, "Y_flat", Y_flat.shape)

        X_reg = X_flat.reshape((tinit * num_nodes, args.w))
        Y_reg = Y_flat.reshape(tinit * num_nodes)
        # print('X_reg'.X_reg.shape, 'Y_reg',Y_reg.shape)
        X_reg = X_reg.cpu().numpy()
        Y_reg = Y_reg.cpu().numpy()

        model = LinearRegression(fit_intercept=False)
        model.fit(X_reg, Y_reg)
        alpha_estimates = np.array(model.coef_)
        alpha_estimates = alpha_estimates[::-1]

        AR = np.zeros_like(Y_t)
        # print("Y_t: ", Y_t.shape)
        # print('Y_t', Y_t[:, :5])

        # alpha_estimates = [0.7, 0.3]
        for k in range(1, args.w + 1):
            AR[:, args.w:] += alpha_estimates[k - 1] * Y_t[:, args.w - k: t_num - k]
            # print(Y_t[-1, args.w - k: t_num - k][-3:])
            # print(AR[-1, args.w:][-3:])

        residuals = (Y_t - AR)[:, args.w:]
        # print("residuals: ", residuals.shape)
        # print('residuals', residuals[:,5:])
    else:
        residuals = Y_t
        alpha_estimates = []

    residuals_centered = residuals - np.mean(residuals, axis=1, keepdims=True)
    C_empirical = (residuals_centered @ residuals_centered.T) / (t_num - args.w)

    dist_matrix = args.D
    # print('D_arr: ',dist_matrix, dist_matrix.shape)

    dataset = args.dataset
    if dataset == 'syn_tailup_gen':
        G =  read_G(args.syn_seed)
        weights_dict = nx.get_edge_attributes(G, "weight")
        num_sampled_obs_per_edge = args.num_sampled_obs_per_edge

        from data.gen_syn_tailup import sort_edges_by_preceding_order
        sorted_edges = sort_edges_by_preceding_order(G)
        weights = []
        for edge in sorted_edges:
            weights.extend([weights_dict[edge] for _ in range(num_sampled_obs_per_edge)])
        # print(f"alpha Estimation: {alpha_estimates},\n Ground-truth: {args.alpha_true}")
    else:
        weights = None
    initial_guess = [1.0, 1.0]

    result = least_squares(opt_objective, initial_guess, args=( C_empirical, dist_matrix, args.num_nodes, weights, dataset), bounds=(1e-5, np.inf))
    estimated_beta, estimated_phi = result.x

    # print(f"sigma Estimation: {estimated_beta}, Ground-truth: {args.beta_true}")
    # print(f"phi Estimation: {estimated_phi}, Ground-truth: {args.phi_true}")

    cov_spatial = build_spatial_covariance(dataset, estimated_beta, estimated_phi, args.num_nodes, weights, dist_matrix)

    return alpha_estimates, residuals, cov_spatial


def sample_cov_from_residual(residuals, device):
    eps_res = torch.Tensor(residuals.T).to(device)
    mean = torch.mean(eps_res, dim=0)
    eps_centered = eps_res - mean
    cov_sample = eps_centered.T @ eps_centered / (eps_centered.shape[0] - 1)
    return cov_sample, eps_centered, mean


def cp_square( model, data_loader, alpha, args ):
    y_true=[]
    x=[]
    with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                y_true.append(label)
                x.append(data)

    X = torch.cat(x,dim=0).squeeze(-1)
    y_true = torch.cat(y_true,dim=0)

    T = y_true.shape[0]
    d = y_true.shape[2]

    tinit = args.tinit
    intervals = []
    volumes = []
    coverage = []
    Y_pred_all = []

    # Ypred = model.predict(X)
    for t in range(tinit, T):
        # Calibration dataset: recent points in the range [t-tinit, t)
        YCal = y_true[t - args.tinit:t]  # ytrue: 500，5，1
        alpha_estimates, residuals, _ = get_cov_spatial(X, t, args.tinit, YCal, args)
        # calibration_Y = Y[t - tinit:t]
        # calibration_predictions = Ypred[t - tinit:t]

        # Compute residuals (scores) for each dimension
        calibration_scores = torch.abs( torch.Tensor(residuals) ) # with shape tinit*d
        # Quantiles for each dimension (1-alpha/d)
        b = np.ceil((tinit + 1) * (1 - alpha / d)) / tinit
        b = np.min((b,1))
        quantiles = torch.quantile(calibration_scores, b, axis=1)
        # print('calibration_scores', calibration_scores.shape, 'quantiles',quantiles.shape)
        # print('y_true', y_true.shape)

        Y = y_true.squeeze(dim=1).squeeze(dim=-1)

        # Predict at time t
        if args.w:
            y_true_t = Y[t].squeeze(-1)
            y_pred_t = torch.zeros_like(y_true_t)
            for k in range(1, args.w + 1):
                y_pred_t += alpha_estimates[k - 1] * Y[t - k].squeeze(-1)
            Y_pred_all.append(y_pred_t)
            # print('y_pred_t',y_pred_t, 'y_true_t', y_true_t)
        else:
            y_true_t = Y[t].squeeze(-1)
            y_pred_t = torch.zeros_like(y_true_t)
            # eps_t = y_true[t].squeeze(-1)


        # Build prediction intervals for each dimension
        interval_t = []
        volume_t = 1
        in_interval = True

        for dim in range(d):
            lower_bound = y_pred_t[dim] - quantiles[dim]
            upper_bound = y_pred_t[dim] + quantiles[dim]
            interval_t.append( (lower_bound, upper_bound) )

            # Update volume
            volume_t *= (upper_bound - lower_bound)

            # Check if true value falls within the interval
            if not (lower_bound <= Y[t, dim] <= upper_bound):
                in_interval = False
        volume_t1 = volume_t ** (1 / d)
        intervals.append(interval_t)
        volumes.append(volume_t1)
        coverage.append(in_interval)
    picp, ineff = np.mean(coverage), np.mean(volumes)
    eff_var =  np.var(volumes)
    print(f"average of picp is {picp} and inefficiency is {ineff} with square cp")
    if args.w > 0:
        Y_pred_all = torch.stack(Y_pred_all)
    return Y_pred_all, picp, ineff, eff_var #, picp, ineff

def PCP_ellip(model, data_loader, alpha, args):

    y_true=[]
    x=[]
    with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                y_true.append(label)
                x.append(data)
    x=torch.cat(x,dim=0)
    y_true=torch.cat(y_true,dim=0)

    T = len(y_true)
    alphatlist=[]
    adaptErrSeq = torch.zeros((T - args.tinit))
    alphat = alpha

    Y_pred_all = []
    eff_l = []

    for t in range(args.tinit, T):

        #### step 1: calculate cov_spatial ####
        YCal = y_true[t - args.tinit:t]  # ytrue: 500，5，1

        alpha_estimates, residuals, cov_spatial = get_cov_spatial(x, t, args.tinit, YCal, args)
        # print(alpha_estimates)

        #### step 2: calculate cov_sample ####
        cov_sample, eps_centered, mean = sample_cov_from_residual(residuals, args.device)
        # print('cov_sample', cov_sample)
        # sample_err = compare_gt_error(cov_sample, args)
        # print('Sample Cov & GT l1 Error: ', sample_err)

        # print("cov_spatial", cov_spatial)
        # print("cov_sample", cov_sample)

        ####  step 3: combine cov_sample & cov_spatial ####

        assert 0 <= args.lmbd <= 1, f"Variable x={args.lmbd} is not in the range [0, 1]"

        if args.Cov_type == 'ellip' :
            cov_sample_inv = torch.linalg.pinv(cov_sample.to(args.device))
            cov_spatial_inv = torch.linalg.pinv(torch.Tensor(cov_spatial).to(args.device))
            cov_inv = (1 - args.lmbd) * cov_sample_inv + args.lmbd * cov_spatial_inv
            cov_matrix = torch.linalg.pinv(cov_inv)

            # cov_matrix = (1-args.lmbd) * cov_sample.to(args.device) +  args.lmbd * torch.Tensor(cov_spatial).to(args.device)
            # cov_inv= torch.linalg.pinv(cov_matrix).to(args.device)

        elif args.Cov_type == 'sphere':
            cov_matrix = torch.eye(args.num_nodes).to(args.device)
            cov_inv = torch.linalg.pinv(cov_matrix)

        else: # args.Cov_type == 'GT':
            if args.dataset == 'gen_tailup':
                from model.BasicTrainer import convert_str_2_tensor
                newshape = (args.num_nodes, args.num_nodes)
                cov_matrix = convert_str_2_tensor(args.Sigma_spatial, newshape, args.device).float()
            else: #'gen_tailip_syn'
                path_gt_cov = './data/syn_data/gt_cov_{}.npz'.format(args.syn_seed)
                A = read_npz(path_gt_cov)
                cov_gt = A + A.T - np.diag(np.diag(A))
                cov_matrix = torch.Tensor(cov_gt)
            cov_matrix = cov_matrix.to(args.device)
            cov_inv = torch.linalg.pinv(cov_matrix)

        # Compute nonconformity scores for the calibration set
        # print(eps_centered.shape, cov_inv.shape,eps_centered.shape )
        n = args.tinit - args.w
        distances = torch.sqrt(torch.sum(eps_centered @ cov_inv * eps_centered, dim=1)).view( n, args.K)   # Shape: (K,)

        # plt.plot(distances.cpu().numpy())
        # plt.savefig('test_dis.png')

        nonconformity_scores,_ = torch.min(distances,dim=1)

        # Convert nonconformity scores to a tensor
        nonconformity_scores = torch.tensor(nonconformity_scores)
        alphat = np.clip(alphat, 0, 1)
        a = torch.tensor(1 - alphat, device = nonconformity_scores.device )
        b = torch.ceil(( n + 1) * a) / n
        b = torch.clip(b, 0, 1).to(nonconformity_scores.dtype)
        # print("the shape of b is {}".format(b.shape))
        q_alpha = torch.quantile(nonconformity_scores, b, interpolation='higher')
        # print('q_alpha', q_alpha)

        # Get predictions for the t-th test sample
        #y_pred_t = y_pred[t].squeeze(-1)  # Shape: (K, 5, 1)
        # AR[:, args.w:] += alpha_estimates[k - 1] * Y_t[:, args.w - k: t_num - k]

        if args.w:
            y_true_t = y_true[t].squeeze(-1)
            y_pred_t = torch.zeros_like(y_true_t)
            for k in range(1, args.w + 1):
                y_pred_t += alpha_estimates[k - 1] * y_true[t-k].squeeze(-1)
            Y_pred_all.append(y_pred_t.squeeze(-1))
            eps_t=y_pred_t-y_true_t
        else:
            Y_pred_all = None
            eps_t = y_true[t].squeeze(-1)
        # print('residuals', torch.Tensor(residuals.T), 'eps_t', eps_t)
        eps_t_centered = eps_t - mean
        # Compute conformity for the t-th test sample
        distances = torch.sqrt(torch.sum(eps_t_centered @ cov_inv * eps_t_centered, dim=1))
        dist=torch.min(distances)

        # print('eps_t',eps_t, 'dist', dist, 'q_alpha',q_alpha)
        adaptErrSeq[t - args.tinit] = (dist > q_alpha).float() # Coverage error
        alphat += args.gamma * (alpha - adaptErrSeq[t - args.tinit])
        # print('cover: ', adaptErrSeq[t - args.tinit] )
        # Store the adaptive alpha at time t
        alphatlist.append(alphat)

        cov_np = cov_matrix.cpu().numpy()
        q_np=q_alpha.cpu().numpy()
        # print('q_np', q_np)

        if args.Cov_type == 'ellip' or args.Cov_type == 'GT':
            cur_eff = ( ellipsoid_volume(cov_np,q_np) )** (1/args.num_nodes)
        else: #elif: args.Cov_type == 'sphere':
            cur_eff = ( sphere_volume(args.num_nodes, q_alpha) ) #** (1/args.num_nodes)
        eff_l.append(cur_eff)

        # print('nonconformity_scores ',nonconformity_scores)
        # print('q_alpha',q_alpha,'dist',dist)
        # print(t - args.tinit, 'cover:', adaptErrSeq[t - args.tinit], 'cur_eff: ', cur_eff)

    picp = 1 - torch.sum(adaptErrSeq) / len(adaptErrSeq)
    eff = np.mean(eff_l)
    eff_var = np.var(eff_l)
    print(f"average of picp is {picp} and inefficiency is {eff} with {args.Cov_type}")
    if args.w > 0:
        Y_pred_all = torch.stack(Y_pred_all)
    return Y_pred_all, picp, eff, eff_var

def read_npz(path):
    loaded_data = np.load(path)
    data = loaded_data['array']
    return data

def PCP_ellip_nonlinear(model, data_loader,alpha,args):
    """
    Use ANY prediction model instead of linear regressor.
    :param model: Prediction Model.
    :param data_loader: Test data loader.
    :param alpha: Desired miscoverage rate.
    :return:
    """
    y_true=[]
    y_pred=[]
    x=[]
    #model.to(args.device)
    with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                temp=[]
                for k in range(args.K):
                    # output = model.predict(data, target, teacher_forcing_ratio=0)
                    output,_ = model.forward(data )
                    temp.append(output)
                temp1=torch.cat(temp,dim=1)
                y_pred.append(temp1)
                y_true.append(label)
                x.append(data)
    # print('data', data.shape , 'output', output.shape, 'temp1',temp1.shape)
    x=torch.cat(x,dim=0) #shape[3988,12,5,1]
    y_pred=torch.cat(y_pred,dim=0) #shape[3988,K,5,1]
    y_true=torch.cat(y_true,dim=0)#shape[3988,1,5,1]

    if args.scaler:
        y_pred = args.scaler.inverse_transform(y_pred)
        y_true = args.scaler.inverse_transform(y_true)
    Y_pred_all = y_pred.squeeze(-1) [args.tinit:]
    # print("Y_pred_all", Y_pred_all.shape)

    num_node=y_true.shape[2]

    T_val = int( len(y_true) * args.val_ratio/ (args.val_ratio + args.test_ratio) )
    T_test = int ( len(y_true) * args.test_ratio / (args.val_ratio + args.test_ratio) )
    print('T_val',T_val,'T_test',T_test)

    alphatlist=[]
    adaptErrSeq =torch.zeros((T_test - args.tinit))
    alphat = alpha
    eff=0
    eff_l = []
    print('test:',  T_test - args.tinit )

    for t in range(args.tinit + T_val, T_val + T_test):
        if args.model == 'A3TGCN':
            y_true = y_true.squeeze(1)

        predcal, YCal = y_pred[t-args.tinit:t], y_true[t - args.tinit:t]  # calibration set y_pred 500，k，5，1 ytrue: 500，5，1
        # predcal, YCal = y_pred[:t], y_true[ :t]

        eps=predcal-YCal
        # print(predcal.shape, 'YCal', YCal.shape,'eps',eps.shape)

        eps_res = eps.view(-1,num_node).T
        # print('eps_res: ', eps_res.shape, 'y_true',y_true.shape)
        mean = torch.mean(eps_res, dim=1, keepdim=True)  # 均值形状为 (1, 5)

        eps_centered= (eps_res-mean).to(args.device)#500*K，5
        # print('eps_centered', eps_centered.shape)


        dataset = args.dataset

        ### ###
        # if t == args.tinit + T_val :
        cov_sample = (eps_centered @ eps_centered.T) / (eps_centered.shape[0] - 1)

        # print('cov_sample', cov_sample.shape)

        dist_matrix = args.D
        C_empirical = cov_sample.cpu().numpy()
        # print(C_empirical)

        if args.weight_type == 'fixed':
            weights = [1 for _ in range(args.num_nodes)]
            #weights = [1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4]
            initial_guess = [1.0, 1.0]
            result = least_squares(opt_objective, initial_guess,
                                   args=(C_empirical, dist_matrix, args.num_nodes, weights, dataset),
                                   bounds=(1e-5, 1e5))
            estimated_beta, estimated_phi = result.x
        elif args.weight_type == 'estimation':
            initial_guess = [1.0] * (args.num_nodes + 2)
            result = least_squares(opt_objective_weights, initial_guess,
                                   args=(C_empirical, dist_matrix, args.num_nodes, dataset),
                                   bounds=(1e-5, 1e5))
            estimated_beta, estimated_phi = result.x[0], result.x[1]
            weights = result.x[1:]
        elif args.weight_type == 'offline':
            if t == args.tinit + T_val :
                weights = [1 for _ in range(args.num_nodes)]
                initial_guess = [1.0, 1.0]
                result = least_squares(opt_objective, initial_guess,
                                       args=(C_empirical, dist_matrix, args.num_nodes, weights, dataset),
                                       bounds=(1e-5, 1e5))
                estimated_beta, estimated_phi = result.x
        else: # 'history'
            his_y = YCal.squeeze(1).squeeze(-1)
            # print(his_y.shape)
            his_mean = torch.mean(his_y, axis = 0)
            # print(his_mean[0].item() )
            weights = his_mean
            w1, w2 , w3 = his_mean[1].item(), his_mean[6].item(), his_mean[0].item()
            weights[0] = 1.0
            weights[8:] *= 1 / w3
            # print( w1, w2 , w3 )
            weights[1:8] *= 1 / (w1+w2)

            # print('weights', weights )

            initial_guess = [1.0,10]
            result = least_squares(opt_objective, initial_guess,
                                   args=(C_empirical, dist_matrix, args.num_nodes, weights, dataset),
                                   bounds=(1e-5, np.inf))
            # print(C_empirical)
            estimated_beta, estimated_phi = result.x
        # estimated_paras = estimated_beta, estimated_phi

        cov_spatial = build_spatial_covariance(dataset, estimated_beta, estimated_phi, args.num_nodes, weights,dist_matrix)
        # print('cov_spatial:', cov_spatial)
        # print('estimated_beta', estimated_beta, 'estimated_phi',estimated_phi)
        # print('cov_sample', cov_sample, 'cov_spatial', cov_spatial)
        if args.Cov_type == 'ellip':
            cov_spatial = torch.Tensor( cov_spatial)
            cov_sample_inv = torch.linalg.pinv(cov_sample.to(args.device))
            cov_spatial_inv = torch.linalg.pinv(torch.Tensor(cov_spatial).to(args.device))
            cov_inv = (1 - args.lmbd) * cov_sample_inv + args.lmbd * cov_spatial_inv
            cov_matrix = torch.linalg.pinv(cov_inv)
        else: #args.Cov_type == 'sphere':
            cov_matrix = torch.eye(args.num_nodes).to(args.device)
            cov_inv = torch.eye(args.num_nodes).to(args.device)

        eps_centered = eps_centered.T
        # print('eps_centered',eps_centered.shape,'cov_inv',cov_inv.shape)
        # Compute nonconformity scores for the calibration set
        distances = torch.sqrt(torch.sum(eps_centered @ cov_inv * eps_centered, dim=1)).view(args.tinit, args.K) #.view(args.tinit+T_val, args.K)   # Shape: (K,)
        #print(distances.shape)

        nonconformity_scores,_ = torch.min(distances,dim=1)
        non_nan_values = nonconformity_scores[~torch.isnan(nonconformity_scores)]
        max_value = torch.max(non_nan_values) if len(non_nan_values) > 0 else torch.tensor(0.0)
        nonconformity_scores = torch.where(torch.isnan(nonconformity_scores), max_value, nonconformity_scores)

        # nonconformity_scores = torch.tensor(nonconformity_scores)
        alphat = np.clip(alphat, 0, 1)
        a = torch.tensor(1 - alphat, device = nonconformity_scores.device )
        b = torch.ceil((args.tinit + 1) * a) / args.tinit
        b = torch.clip(b, 0, 1).to(nonconformity_scores.dtype)
        # print("the shape of b is {}".format(b.shape))
        q_alpha = torch.quantile(nonconformity_scores, b, interpolation='higher')
        # print('q_alpha ', q_alpha)

        # Get predictions for the t-th test sample
        y_pred_t = y_pred[t- T_val].squeeze(-1)  # Shape: (K, 5, 1)
        y_true_t = y_true[t - T_val].squeeze(-1)  # Shape: (5, 1)
        eps_t= (y_pred_t-y_true_t).squeeze(0)
        # print('eps_t', eps_t.shape, 'mean', mean.shape)
        eps_t_centered=eps_t-mean.squeeze(0)

        # Compute conformity for the t-th test sample
        distances = torch.sqrt(torch.sum(eps_t_centered @ cov_inv * eps_t_centered, dim=1))
        dist=torch.min(distances)

        adaptErrSeq[t - T_val - args.tinit] = (dist > q_alpha).float() # Coverage error
        # if (dist > q_alpha).float():
        #     print('Not covered')
        # print('dist', dist, 'q_alpha', q_alpha)

        alphat += args.gamma * (alpha - adaptErrSeq[t - T_val - args.tinit])
        # Store the adaptive alpha at time t
        alphatlist.append(alphat)
        cov_np=cov_matrix.cpu().numpy()
        q_np=q_alpha.cpu().numpy()
        # print('q_alpha', q_alpha)

        if args.Cov_type == 'ellip':
            cur_eff = (args.K * ellipsoid_volume(cov_np, q_np)) ** (1 / args.num_nodes)
        else:  # elif: args.Cov_type == 'sphere':
            cur_eff = (args.K * sphere_volume(args.num_nodes, q_alpha)) #** (1 / args.num_nodes)

        # if t % 1000 == 0:
        # # if torch.isnan(q_alpha).any():
        #     print('estimated_beta, estimated_phi ', estimated_beta, estimated_phi )
        #     print('cov_sample', cov_sample)
        #     print('cov_spatial', cov_spatial)
        #     print('cov_inv', cov_inv)
        #     print('eps_centered', eps_centered)
        #     print('nonconformity_scores', nonconformity_scores)
        #     print(np.linalg.eigvals(cov_np))
        #     # nan_indices = torch.nonzero(torch.isnan(nonconformity_scores), as_tuple=True)
        #     # print( nan_indices, eps_centered[nan_indices] )
        #     # print( torch.sqrt(torch.sum( eps_centered[nan_indices] @ cov_inv * eps_centered[nan_indices])) )

        # if t % 1000 == 0:
        #     print(t, 'cur_eff', cur_eff, 'q_alpha', q_alpha)

        eff_l.append(cur_eff)
        # eff += cur_eff / (T-args.tinit)


    picp = 1 - torch.sum(adaptErrSeq) / len(adaptErrSeq)
    eff = np.nanmean(eff_l)
    eff_var = np.var(eff_l)
    print(f"average of picp is {picp} and inefficiency is {eff}")
    print('Num of nan', np.sum(np.isnan(eff_l))  )
    return Y_pred_all, picp, eff, eff_var