import numpy as np

import torch
import scipy.special 
import math
import alphashape
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import torch.nn.functional as F

def score_L2(vector1, vector2):
    diff = (vector1 - vector2) ** 2
    if diff.dim() >1:
        sum_squared_diff = torch.sum(diff, dim=1)
    else:
        sum_squared_diff = torch.sum(diff)

    # Compute the square root to get the Euclidean distance
    distances = torch.sqrt(sum_squared_diff)

    return distances
def generate_points_within_boundary(num_points, bounds):
    lower_bounds = torch.floor(bounds[0].squeeze())
    upper_bounds = torch.ceil(bounds[1].squeeze())
    lengths=upper_bounds-lower_bounds
    volume=torch.prod(lengths)
    
    # Generate random points in [0, 1] for each dimension and scale to bounds
    # random_points = torch.from_numpy(np.random.uniform(0, 1, (num_points, len(bounds[0])))).to(lower_bounds.device)
    random_points = (torch.rand((num_points, len(bounds[0])), device=lower_bounds.device))
    points = lower_bounds + random_points * (upper_bounds - lower_bounds)
    return points,volume

def aci_map_graph(ypred, ytrue, alpha, gamma, device, correctionmodel, gap_m, tinit=300, link_pred =True):

    ytrue = ytrue.to(device) if not ytrue.is_cuda else ytrue  # ytrue timepoints*num_nodes
    ypred = ypred.to(device) if not ypred.is_cuda else ypred
    # print(ytrue.shape)
    correctionmodel.to(device)
    correctionmodel.eval()
    # mapypred: prediction mapped at low dimensional space
    # ypred: original prediction
    with torch.no_grad():
        mapypred = correctionmodel(ypred)
        mapytrue = correctionmodel(ytrue)
    T = len(ytrue)  # ytrue: timepoints*12*307*1
    adaptErrSeq = torch.zeros(T - tinit)
    alphat = alpha
    mapytrue, mapypred = mapytrue[0], mapypred[0]

    ### picp ###
    min_vals = torch.min(ytrue.squeeze().min(dim=0).values, ypred.squeeze().min(dim=0).values)
    max_vals = torch.max(ytrue.squeeze().max(dim=0).values, ypred.squeeze().max(dim=0).values)
    bounds = torch.stack([min_vals, max_vals], dim=0)
    vol=[]
    volsim=[]
    ######

    print(T,"timestep in total...")
    for t in range(tinit, T):

        # YCal, predCal = ytrue[t - tinit:t], ypred[t - tinit:t]  # calibration set
        mapYCal, mappredCal = mapytrue[t - tinit:t] ,  mapypred[t - tinit:t]
        L2_loss = score_L2(mapYCal, mappredCal)

        if not link_pred:
            graph_loss, err_link = 0, 0
        else:
            graph_loss = torch.sum( torch.abs(gap_m[t-tinit:t]), dim = (1,2))
            err_link = torch.sum(torch.abs(gap_m[t]))

            min_val = torch.min(graph_loss)
            max_val = torch.max(graph_loss)
            graph_loss = (graph_loss - min_val) / (max_val - min_val) #*  torch.max(L2_loss)
            err_link = (err_link - min_val) / (max_val - min_val) #* torch.max(L2_loss)

        #print('graph_loss    ', graph_loss.shape)
        # print('score    ',score_L2(mapYCal, mappredCal).shape)

        scores = L2_loss + graph_loss  # shape tinit*num_nodes
        # scores = L2_loss * graph_loss

        # print( 'score_L2', score_L2(mapYCal, mappredCal)[:10], graph_loss[:10])

        calscores = scores.reshape(tinit, -1)
        alphat = np.clip(alphat, 0, 1)
        # print("the shape of alpha is {}".format(alphat.shape))
        # print("the shape of scores is {}".format(calscores.shape))
        # print("the type of alpha is {}".format((1 - alpha_tensor).dtype))
        a = torch.tensor(1 - alphat, device = calscores.device )
        b = torch.ceil((tinit + 1) * a) / tinit
        b = torch.clip(b, 0, 1).to(calscores.dtype)
        # print("the shape of b is {}".format(b.shape))
        qu = torch.quantile(calscores, b, interpolation='higher')
        


        err_score_l2 = score_L2(mapypred[t], mapytrue[t])
        err_t = err_score_l2 + err_link
        # err_t = err_score_l2 * err_link


        errSeq = 1 - (qu >=  err_t).float()
        #print(errSeq)

        ### picp ### ??????????
        simu_point,bound_vol= generate_points_within_boundary(1000000,bounds)
        simu_pred_map = correctionmodel( simu_point) [0]
        scores = score_L2( simu_pred_map, mapytrue[t].expand_as(simu_pred_map))

        prob = (scores + err_link <= qu).float()
        # prob = (scores * err_link <= qu).float()

        # prob=[0 if score_L2(  correctionmodel( simu_point[i] ).unsqueeze(0), mapytrue[t]  )
        #            +err_link >qu else 1 for i in range(len(simu_point))]
        vol_simu= torch.mean(prob)
        volsim.append(vol_simu)
        #########

        # print("the shape of errSeq is {}".format(errSeq.shape))
        adaptErrSeq[t - tinit] = errSeq
        # print("the shape of adaperrSeq is {}".format(adaptErrSeq.shape))

        # print('before',alphat, errSeq, 'with', gamma)
        alphat += gamma * (alpha - adaptErrSeq[t - tinit])
        '''
        if t and t % 500 == 0:
            print(t, "in progress...")
            print('vol_simu', vol_simu, 'errSeq', errSeq, 'alphat', alphat)
        '''
        # 计算上下界

        del mapYCal, mappredCal, scores  # 释放缓存
        torch.cuda.empty_cache()

    picp = 1 - torch.sum(adaptErrSeq) / len(adaptErrSeq)
    eff = sum(volsim) / len(volsim)
    print(f"average of picp is {picp} and inefficiency is {eff}")

    return picp, eff


def aci_graph(ypred, ytrue, alpha, gamma,device,gap_m, tinit=300, link_pred =True):

    ytrue = ytrue.to(device) if not ytrue.is_cuda else ytrue
    ypred = ypred.to(device) if not ypred.is_cuda else ypred
    ytrue=ytrue.squeeze()
    ypred=ypred.squeeze()

    min_vals=torch.min(ytrue.min(dim=0).values,ypred.min(dim=0).values)
    max_vals=torch.max(ytrue.max(dim=0).values,ypred.max(dim=0).values)
    bounds=torch.stack([min_vals,max_vals],dim=0)
    T = len(ytrue)
    volsim = []

    alphatlist=[]
    adaptErrSeq = np.zeros((T - tinit))
    alphat = alpha

    for t in range(tinit, T):
        
        YCal, predCal = ytrue[t - tinit:t], ypred[t - tinit:t]  # calibration set

        graph_loss, err_link = 0, 0
        if link_pred:
            graph_loss = torch.sum( torch.abs(gap_m[t-tinit:t]), dim = (1,2))
            err_link = torch.sum(torch.abs(gap_m[t]))

            min_val = torch.min(graph_loss)
            max_val = torch.max(graph_loss)
            graph_loss = (graph_loss - min_val) / (max_val - min_val)
            err_link = (err_link - min_val) / (max_val - min_val)

        scores = score_L2(YCal,predCal) + graph_loss # shape tinit*num_nodes
        # scores = score_L2(YCal, predCal) * graph_loss

        calscores=scores.reshape(tinit,-1)
        alphat = np.clip(alphat, 0, 1)

        a=(1 - alphat)
        b=torch.ceil(torch.tensor((tinit+1)*a))/tinit
        b=torch.clip(b,0,1).to(calscores.device)
        #print("the shape of b is {}".format(b.shape))
        alphatlist.append(1-b)
        qu = torch.quantile(calscores, b.to(calscores.dtype), interpolation='higher')

        simu_point, bound_vol = generate_points_within_boundary(1000000, bounds)
        err_simu = score_L2(simu_point, ypred[t].expand_as(simu_point))

        prob = (err_simu + err_link <= qu).float()
        # prob = (err_simu * err_link <= qu).float()

        vol_simu = torch.mean(prob)
        volsim.append(vol_simu)

        # volume=(math.pi ** (5 / 2) * qu ** 5) / scipy.special.gamma(5 / 2 + 1)
        # #print('the volume of the bounding box is{}'.format(bound_vol))
        # #print("the volume is{} and the simulated volum is {}".format(volume,vol_simu))
        # vol.append(volume)

        #print("the shape of q is {}".format(q.shape))
        # 计算adjusted error sequence
        err_score_l2 = score_L2( ypred[t].unsqueeze(0), ytrue[t].unsqueeze(0) )
        err_t = err_score_l2 + err_link
        # err_t = err_score_l2 * err_link
        errSeq = 1 - (qu >= err_t).float()
        #print("the shape of errSeq is {}".format(errSeq.shape))
        adaptErrSeq[t - tinit] = errSeq.cpu().numpy()
        #print("the shape of adaperrSeq is {}".format(adaptErrSeq.shape))

        alphat += gamma * (alpha - adaptErrSeq[t - tinit])
        '''  
        if  t%500==0:
            print(t, "in progress...")
            print('vol_simu', vol_simu, 'errSeq', errSeq, 'alphat',alphat )
        '''
        del YCal, predCal, scores
        torch.cuda.empty_cache()


    picp = 1-np.sum(adaptErrSeq)/len(adaptErrSeq)
    eff = sum(volsim)/len(volsim)
    print(f"average of picp is {picp} and inefficiency is {eff}")
    print(sum(alphatlist)/len(alphatlist))

    return picp, eff
