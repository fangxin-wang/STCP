
import numpy as np

import torch
import scipy.special 
import math
import alphashape
import matplotlib.pyplot as plt

import torch.nn.functional as F

from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize, least_squares
from data.gen_syn_graph_tailup import opt_objective, build_spatial_covariance

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


def get_cov_spatial(x, t, YCal, args):
    num_nodes = args.num_nodes

    X = x[t - args.tinit:t, -args.w:, :].squeeze()
    X = X.permute(0, 2, 1)
    X_flat = X
    # X_flat = X.reshape(X.size(0), -1)
    Y_flat = YCal.squeeze()
    # print(X_flat[-3:,-1,:], Y_flat[-3:,-1])

    # X_flat, Y_flat = X_flat[args.w : ], Y_flat[args.w : ]
    # print("X_flat", X_flat.shape, "Y_flat", Y_flat.shape)

    X_reg = X_flat.reshape((args.tinit * num_nodes, args.w))
    Y_reg = Y_flat.reshape(args.tinit * num_nodes)
    # print('X_reg'.X_reg.shape, 'Y_reg',Y_reg.shape)
    X_reg = X_reg.cpu().numpy()
    Y_reg = Y_reg.cpu().numpy()

    model = LinearRegression(fit_intercept=False)
    model.fit(X_reg, Y_reg)
    alpha_estimates = np.array(model.coef_)
    alpha_estimates = alpha_estimates[::-1]

    Y_t = np.array(Y_flat.cpu().numpy()).T
    t_num = Y_t.shape[1]

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

    residuals_centered = residuals - np.mean(residuals, axis=1, keepdims=True)
    C_empirical = (residuals_centered @ residuals_centered.T) / (t_num - args.w)
    # print('C_empirical: ',C_empirical)

    D_arr = args.D
    #D_arr = D_arr.cpu().detach().numpy()
    # print('D_arr: ',D_arr)

    initial_guess = [1.0, 1.0]
    result = least_squares(opt_objective, initial_guess, args=(D_arr, C_empirical), bounds=(1e-5, np.inf))
    estimated_sigma2, estimated_phi = result.x

    # print(f"alpha Estimation: {alpha_estimates},\n Ground-truth: {args.alpha_true}")
    # print(f"sigma Estimation: {estimated_sigma2},\n Ground-truth: {args.sigma2_true}")
    # print(f"phi Estimation: {estimated_phi},\n Ground-truth: {args.phi_true}")

    cov_spatial = build_spatial_covariance(D_arr, estimated_sigma2, estimated_phi)

    return alpha_estimates, residuals, cov_spatial

def PCP_ellip(model, data_loader, alpha, args):
    y_true=[]
    y_pred=[]
    x=[]
    #model.to(args.device)
    with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                temp=[]
                # for _ in range(args.K):
                #     # output = model.predict(data, target, teacher_forcing_ratio=0)
                #     output,_ = model.forward(data, None, teacher_forcing_ratio=0)
                #     temp.append(output)
                temp1=torch.cat(temp,dim=1)
                y_pred.append(temp1)
                y_true.append(label)
                x.append(data)

    x=torch.cat(x,dim=0) #shape[3988,12,5,1]
    y_pred=torch.cat(y_pred,dim=0) #shape[3988,K,5,1]
    y_true=torch.cat(y_true,dim=0)#shape[3988,1,5,1]

    # print('y_true', y_true[:, -5:])

    # print(y_pred.shape, y_true.shape)
    # print("====", y_true[-10:, :], "==========")
    num_node=y_true.shape[2]

    T = len(y_true)
    alphatlist=[]
    adaptErrSeq = torch.zeros((T - args.tinit))
    alphat = alpha
    eff=0
    Y_pred_all = []

    for t in range(args.tinit, T):

        #### step 1: calculate cov_spatial ####
        YCal = y_true[t - args.tinit:t]  # ytrue: 500，5，1
        alpha_estimates, residuals, cov_spatial = get_cov_spatial(x, t, YCal, args)

        #### step 2: calculate cov_sample ####
        #print(predcal.shape)
        # eps = YCal - predcal #500,K,5,1
        # eps_res=eps.view(-1,num_node) # torch.Size([300, 6])

        eps_res = torch.Tensor(residuals.T).to(args.device)
        mean = torch.mean(eps_res, dim=0)


        eps_centered=eps_res-mean #500*K，5
        # print(eps_centered.shape)
        cov_sample = eps_centered.T @ eps_centered / (eps_centered.shape[0] - 1)  # 公式：X^T X / (n-1)
        #
        # print("cov_spatial", cov_spatial)
        # print("cov_sample", cov_sample)

        ####  step 3: combine cov_sample & cov_spatial ####

        assert 0 <= args.lmbd <= 1, f"Variable x={args.lmbd} is not in the range [0, 1]"

        if args.Cov_type == 'ellip':
            # cov_matrix = (1-args.lmbd) * cov_sample.to(args.device) +  args.lmbd * torch.Tensor(cov_spatial).to(args.device)
            # cov_inv= torch.linalg.pinv(cov_matrix).to(args.device)

            cov_sample_inv = torch.linalg.pinv(cov_sample.to(args.device))
            cov_spatial_inv = torch.linalg.pinv(torch.Tensor(cov_spatial).to(args.device))
            cov_inv = (1 - args.lmbd) * cov_sample_inv + args.lmbd * cov_spatial_inv
            cov_matrix = torch.linalg.pinv(cov_inv)

        elif args.Cov_type == 'sphere':
            cov_matrix = torch.eye(args.num_nodes).to(args.device)
            cov_inv = torch.linalg.pinv(cov_matrix)
        elif args.Cov_type == 'GT':
            from model.BasicTrainer import convert_str_2_tensor
            newshape = (args.num_nodes, args.num_nodes)
            cov_matrix = convert_str_2_tensor(args.Sigma_spatial, newshape, args.device).float()
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

        # Get predictions for the t-th test sample
        #y_pred_t = y_pred[t].squeeze(-1)  # Shape: (K, 5, 1)
        # AR[:, args.w:] += alpha_estimates[k - 1] * Y_t[:, args.w - k: t_num - k]

        y_true_t = y_true[t].squeeze(-1)
        y_pred_t = torch.zeros_like(y_true_t)
        for k in range(1, args.w + 1):
            y_pred_t += alpha_estimates[k - 1] * y_true[t-k].squeeze(-1)
        Y_pred_all.append(y_pred_t.squeeze(-1))

        # Shape: (5, 1)
        eps_t=y_pred_t-y_true_t
        print('residuals', torch.Tensor(residuals.T), 'eps_t', eps_t)
        eps_t_centered=eps_t-mean
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

        if args.Cov_type == 'ellip' or args.Cov_type == 'GT':
            cur_eff = ( args.K * ellipsoid_volume(cov_np,q_np) )** (1/args.num_nodes)
        else: #elif: args.Cov_type == 'sphere':
            cur_eff = (args.K * sphere_volume(args.num_nodes, q_alpha) ) #** (1/args.num_nodes)
        eff += cur_eff

        print('nonconformity_scores ',nonconformity_scores)
        print('q_alpha',q_alpha,'dist',dist)
        print(t - args.tinit, 'cover:', adaptErrSeq[t - args.tinit], 'cur_eff: ', cur_eff)

    picp = 1 - torch.sum(adaptErrSeq) / len(adaptErrSeq)
    print(f"average of picp is {picp} and inefficiency is {eff} with {args.Cov_type}")

    Y_pred_all = torch.stack(Y_pred_all)
    return Y_pred_all, picp,eff

def PCP_ellip_nonlinear(model,data_loader,alpha,args):
    """
    Original Verison with only sample covariance.
    Use ANY prediction model instead of tailup.
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
                    output,_ = model.forward(data, None, teacher_forcing_ratio=0)
                    temp.append(output)
                temp1=torch.cat(temp,dim=1)
                y_pred.append(temp1)

                y_true.append(label)
                x.append(data)
    # print('data', data.shape , 'output', output.shape, 'temp1',temp1.shape)
    x=torch.cat(x,dim=0) #shape[3988,12,5,1]
    y_pred=torch.cat(y_pred,dim=0) #shape[3988,K,5,1]
    y_true=torch.cat(y_true,dim=0)#shape[3988,1,5,1]
    Y_pred_all = y_pred.squeeze(-1) [args.tinit:]
    # print("Y_pred_all", Y_pred_all.shape)

    num_node=y_true.shape[2]

    T = len(y_true)

    alphatlist=[]
    adaptErrSeq =torch.zeros((T - args.tinit))
    alphat = alpha
    eff=0

    for t in range(args.tinit, T):

        predcal, YCal =y_pred[t-args.tinit:t],y_true[t - args.tinit:t]  # calibration set y_pred 500，k，5，1 ytrue: 500，5，1
        # print('predcal', predcal.shape)

        eps=predcal-YCal #500,K,5,1
        eps_res=eps.view(-1,num_node)
        mean = torch.mean(eps_res, dim=0, keepdim=True)  # 均值形状为 (1, 5)

        eps_centered= (eps_res-mean)#500*K，5
        # print('eps_centered', eps_centered.shape)

        if args.Cov_type == 'ellip':
            cov_matrix = eps_centered.T @ eps_centered / (eps_centered.shape[0] - 1)  # 公式：X^T X / (n-1)
            cov_inv= torch.linalg.pinv(cov_matrix).to(args.device)
        else: # args.Cov_type == 'sphere':
            cov_matrix = torch.eye(args.num_nodes).to(args.device)
            cov_inv = torch.eye(args.num_nodes).to(args.device)

        # Compute nonconformity scores for the calibration set
        distances = torch.sqrt(torch.sum(eps_centered @ cov_inv * eps_centered, dim=1)).view(args.tinit, args.K)   # Shape: (K,)
        #print(distances.shape)

        nonconformity_scores,_ = torch.min(distances,dim=1)
        #print(nonconformity_scores.shape)

        # Convert nonconformity scores to a tensor
        nonconformity_scores = torch.tensor(nonconformity_scores)
        alphat = np.clip(alphat, 0, 1)
        a = torch.tensor(1 - alphat, device = nonconformity_scores.device )
        b = torch.ceil((args.tinit + 1) * a) / args.tinit
        b = torch.clip(b, 0, 1).to(nonconformity_scores.dtype)
        # print("the shape of b is {}".format(b.shape))
        q_alpha = torch.quantile(nonconformity_scores, b, interpolation='higher')

        # Get predictions for the t-th test sample
        y_pred_t = y_pred[t].squeeze(-1)  # Shape: (K, 5, 1)
        y_true_t = y_true[t].squeeze(-1)  # Shape: (5, 1)
        eps_t=y_pred_t-y_true_t
        eps_t_centered=eps_t-mean
        # Compute conformity for the t-th test sample
        distances = torch.sqrt(torch.sum(eps_t_centered @ cov_inv * eps_t_centered, dim=1))
        dist=torch.min(distances)

        adaptErrSeq[t - args.tinit] = (dist > q_alpha).float() # Coverage error
        # if (dist > q_alpha).float():
        #     print('Not covered')
        # print('dist', dist, 'q_alpha', q_alpha)

        alphat += args.gamma * (alpha - adaptErrSeq[t - args.tinit])
        # Store the adaptive alpha at time t
        alphatlist.append(alphat)
        cov_np=cov_matrix.cpu().numpy()
        q_np=q_alpha.cpu().numpy()

        if args.Cov_type == 'ellip':
            cur_eff = (args.K * ellipsoid_volume(cov_np, q_np)) ** (1 / args.num_nodes)
        else:  # elif: args.Cov_type == 'sphere':
            cur_eff = (args.K * sphere_volume(args.num_nodes, q_alpha)) #** (1 / args.num_nodes)
        eff += cur_eff


    picp = 1 - torch.sum(adaptErrSeq) / len(adaptErrSeq)

    print(f"average of picp is {picp} and inefficiency is {eff}")
    return Y_pred_all, picp, eff