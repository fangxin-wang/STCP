import numpy as np
import matplotlib.pyplot as plt
import torch
def cp(ytrue, ypred, alpha, device, tinit=500, updateMethod="Simple", momentumBW=0.95):
    T = len(ytrue) #ytrue: timepoints*12*307*1
    #print(ytrue.shape)
    horizon = ytrue.shape[1]
    num_nodes = ytrue.shape[2]
    yup = torch.zeros_like(ypred[:T-tinit]).to(device)
    ylr = torch.zeros_like(ypred[:T-tinit]).to(device)
    q = torch.zeros((horizon, num_nodes, 1), device=device)
    
    for t in range(tinit, T):
        YCal, predCal = ytrue[t-tinit:t], ypred[t-tinit:t]  # the num of calibration timepoints: tinit
        

        scores = torch.abs(YCal - predCal)  # shape tinit*horizon*num_nodes*1
        q=torch.quantile(scores,np.ceil((tinit+1)*(1-alpha))/tinit,dim=0)
        #print(q.shape)

        yup[t-tinit] = ypred[t] + q
        ylr[t-tinit] = ypred[t] - q
    picplist=[]
    mpiwlist=[]
    for time in range (horizon):
        in_num = torch.sum((ytrue[tinit:,time,:] >= ylr[:,time,:]) & (ytrue[tinit:,time,:] <= yup[:,time,:])).item()
        picp = in_num / ytrue[tinit:,time,:].numel()
        mpiw = torch.mean(yup[:,time,:] -  ylr[:,time,:]).item()
        print(f"picp for the time step {time} is {picp} and mpiw is {mpiw}")
        picplist.append(picp)
        mpiwlist.append(mpiw)

    return picplist, mpiwlist  

def cp_cqr(ytrue, ypred,ylowq,yupq, alpha, device, tinit=500):
    T = len(ytrue) #ytrue: timepoints*12*307*1
    #print(ytrue.shape)
    horizon = ytrue.shape[1]
    num_nodes = ytrue.shape[2]
    yup = torch.zeros_like(ypred[:T-tinit]).to(device)
    ylr = torch.zeros_like(ypred[:T-tinit]).to(device)
    q = torch.zeros((horizon, num_nodes, 1), device=device)
    
    for t in range(tinit, T):
        YCal, predCal,up_cal,low_cal = ytrue[t-tinit:t], ypred[t-tinit:t],ylowq[t-tinit:t],yupq[t-tinit:t],  # the num of calibration timepoints: tinit
        

        scores = torch.maximum(YCal-up_cal,low_cal-YCal)
        q=torch.quantile(scores,1-alpha,dim=0)
        #print(q.shape)
        yup[t-tinit] = yupq[t] + q
        ylr[t-tinit] = ylowq[t] - q

    in_num = torch.sum((ytrue[tinit:] >= ylr) & (ytrue[tinit:] <= yup)).item()

    picp = in_num / ytrue[tinit:].numel()
    mpiw = torch.mean(yup - ylr).item()
    print(f"picp is {picp} and mpiw is {mpiw}")

    return picp, mpiw  

def aci(ytrue, ypred, alpha, gamma, device, tinit=500, updateMethod="Simple", momentumBW=0.95):

    ytrue = ytrue.to(device) if not ytrue.is_cuda else ytrue
    ypred = ypred.to(device) if not ypred.is_cuda else ypred

    
    T = len(ytrue) # ytrue: timepoints*12*307*1
    horizon = ytrue.shape[1]
    num_nodes = ytrue.shape[2]


    alphaTrajectory = np.full((T-tinit, num_nodes, horizon), alpha)
    adaptErrSeq = np.zeros((T-tinit, num_nodes, horizon))
    alphat = np.full((num_nodes, horizon), alpha)
    yup = torch.zeros_like(ypred[:T-tinit]).to(device)
    ylr = torch.zeros_like(ypred[:T-tinit]).to(device)
    q = torch.zeros((horizon, num_nodes, 1), device=device)
    
    for t in range(tinit, T):
        YCal, predCal = ytrue[t-tinit:t], ypred[t-tinit:t]  # the num of calibration timepoints: tinit

        scores = torch.abs(YCal - predCal)  # shape tinit*horizon*num_nodes*1
        for i in range(num_nodes):
            for h in range(horizon):
                if alphat[i, h] <= 0:
                    adaptErrSeq[t-tinit, i, h] = 0
                    alphat[i, h] = 0
                elif alphat[i, h] >= 1:
                    adaptErrSeq[t-tinit, i, h] = 1
                    alphat[i, h] = 1
                else:
                    q[h, i, 0] = torch.quantile(scores[:, h, i, :], 1 - alphat[i, h])
                    adaptErrSeq[t-tinit, i, h] = 1 - (q[h, i, 0] >= torch.abs(ypred[t, h, i, 0] - ytrue[t, h, i, 0])).float().item()
                q[h, i, 0] = torch.quantile(scores[:, h, i, :], 1 - alphat[i, h])
        
        alphaTrajectory[t-tinit] = alphat

        if updateMethod == "Simple":
            alphat += gamma * (alpha - adaptErrSeq[t-tinit])

        yup[t-tinit] = ypred[t] + q
        ylr[t-tinit] = ypred[t] - q

        del YCal, predCal, scores  
        torch.cuda.empty_cache() 

    picplist=[]
    mpiwlist=[]
    for time in range (horizon):
        in_num = torch.sum((ytrue[tinit:,time,:] >= ylr[:,time,:]) & (ytrue[tinit:,time,:] <= yup[:,time,:])).item()
        picp = in_num / ytrue[tinit:,time,:].numel()
        mpiw = torch.mean(yup[:,time,:] -  ylr[:,time,:]).item()
        print(f"picp for the time step {time} is {picp} and mpiw is {mpiw}")
        picplist.append(picp)
        mpiwlist.append(mpiw)

    return picplist, mpiwlist  
def aci_cqr(ytrue, ypred,ylowq,yupq, alpha, gamma,device, tinit=500):
    T = len(ytrue) 
    horizon = ytrue.shape[1]
    num_nodes = ytrue.shape[2]
    alphat = np.full((num_nodes, horizon), alpha)
    yup = torch.zeros_like(ypred[:T-tinit]).to(device)
    ylr = torch.zeros_like(ypred[:T-tinit]).to(device)
    q = torch.zeros((horizon, num_nodes, 1), device=device)
    alphaTrajectory = np.full((T-tinit, num_nodes, horizon), alpha)
    adaptErrSeq = np.zeros((T-tinit, num_nodes, horizon))   
    for t in range(tinit, T):
        YCal, predCal,up_cal,low_cal = ytrue[t-tinit:t], ypred[t-tinit:t],ylowq[t-tinit:t],yupq[t-tinit:t],  # the num of calibration timepoints: tinit
        

        scores =torch.maximum(YCal-up_cal,low_cal-YCal)  # shape tinit*horizon*num_nodes*1
        for i in range(num_nodes):
            for h in range(horizon):
                if alphat[i, h] <= 0:
                    adaptErrSeq[t-tinit, i, h] = 0
                    alphat[i, h] = 0
                elif alphat[i, h] >= 1:
                    adaptErrSeq[t-tinit, i, h] = 1
                    alphat[i, h] = 1
                else:
                    q[h, i, 0] = torch.quantile(scores[:, h, i, :], 1 - alphat[i, h])
                    adaptErrSeq[t-tinit, i, h] = 1 - (q[h, i, 0] >= torch.abs(ypred[t, h, i, 0] - ytrue[t, h, i, 0])).float().item()
                    adaptErrSeq[t-tinit, i, h] = 1 - ((ytrue[t, h, i, 0]>=ylowq[t,h,i,0]-q[h,i,0]) and (ytrue[t, h, i, 0]<=yupq[t,h,i,0]+q[h,i,0])).float().item()
                q[h, i, 0] = torch.quantile(scores[:, h, i, :], 1 - alphat[i, h])
        
        alphaTrajectory[t-tinit] = alphat

        
        alphat += gamma * (alpha - adaptErrSeq[t-tinit])

        yup[t-tinit] = yupq[t] + q
        ylr[t-tinit] = ylowq[t] - q

        del YCal, predCal, scores  
        torch.cuda.empty_cache() 




    in_num = torch.sum((ytrue[tinit:] >= ylr) & (ytrue[tinit:] <= yup)).item()

    picp = in_num / ytrue[tinit:].numel()
    mpiw = torch.mean(yup - ylr).item()
    print(f"picp is {picp} and mpiw is {mpiw}")

    return picp, mpiw  

def aci_correction2(ytrue, ypred, alpha, gamma, device,correctionmodel, tinit=320, updateMethod="Simple"):
    # 确保ytrue和ypred是PyTorch张量，并转移到指定设备
    ytrue = ytrue.to(device) if not ytrue.is_cuda else ytrue
    ypred = ypred.to(device) if not ypred.is_cuda else ypred
    correctionmodel.to(device)
    correctionmodel.eval()
    T = len(ytrue)  # ytrue: timepoints*12*307*1
    horizon = ytrue.shape[1]
    num_nodes = ytrue.shape[2]

    alphaTrajectory = np.full((T-tinit, horizon,num_nodes), alpha)
    adaptErrSeq = np.zeros((T-tinit, horizon,num_nodes))
    alphat = np.full((horizon,num_nodes), alpha)
    yup = torch.zeros_like(ypred[:T-tinit]).to(device)
    ylr = torch.zeros_like(ypred[:T-tinit]).to(device)
    q = torch.zeros((horizon, num_nodes, 1), device=device)
    with torch.no_grad():  # 避免在推理时保存梯度
        adjustypred = correctionmodel(ypred.squeeze()).unsqueeze(-1)    
        
    for t in range(tinit, T):

        YCal, predCal = ytrue[t - tinit:t], ypred[t - tinit:t]  # calibration set
        pred = predCal.squeeze()
        with torch.no_grad():  # 避免在推理时保存梯度
            adjust_pred = correctionmodel(pred)
        
        adjust_pred = adjust_pred.view(tinit, predCal.shape[1], predCal.shape[2], 1)

        scores = torch.abs(YCal - adjust_pred)  # shape tinit*horizon*num_nodes*1
        #print("the shape of score is{}".format(scores.shape))
        calscores=scores.reshape(tinit,-1)
        alphat = np.clip(alphat, 0, 1)
        #print("the shape of alpha is {}".format(alphat.shape))
        alpha_tensor = torch.tensor(alphat, device=device).unsqueeze(-1)
        #print("the shape of alpha_tensor is{}".format(alpha_tensor.shape))
        alpha_tensor=alpha_tensor.reshape(-1,1).squeeze()
        #a=(torch.ceil((tinit+1)*(1 - alpha_tensor))/tinit).float()
        
        a=(1 - alpha_tensor).float()
        b=torch.ceil((tinit+1)*a)/tinit
        b=torch.clip(b,0,1)
        #print("the shape of b is {}".format(b.shape))
        qu = torch.quantile(calscores, b, dim=0,interpolation='higher')
    
        #print("the shape of qu is {}".format(qu.shape))
        q=torch.diag(qu).reshape(horizon,num_nodes,1)
        #print("the shape of q is {}".format(q.shape))
        # 计算adjusted error sequence
        errSeq = 1 - (q >= torch.abs(adjustypred[t] - ytrue[t])).float()
        #print("the shape of errSeq is {}".format(errSeq.shape))
        #print("the shape ofadaptErrSeq[t - tinit] is {}".format(adaptErrSeq[t - tinit].shape))
        adaptErrSeq[t - tinit] = errSeq.squeeze().cpu().numpy()
        #print("the shape of adaperrSeq is {}".format(adaptErrSeq.shape))
        # 更新alphat
        #print("the shape of alphat is {}".format(alphat.shape))
        if updateMethod == "Simple":
            alphat += gamma * (alpha - adaptErrSeq[t - tinit])
        

        # 计算上下界
       
  
        yup[t - tinit] = adjustypred[t] + q
        ylr[t - tinit] = adjustypred[t] - q

        del YCal, predCal, scores,adjust_pred,calscores  # 释放缓存
        torch.cuda.empty_cache()

    picplist=[]
    mpiwlist=[]
    for time in range (horizon):
        in_num = torch.sum((ytrue[tinit:,time,:] >= ylr[:,time,:]) & (ytrue[tinit:,time,:] <= yup[:,time,:])).item()
        picp = in_num / ytrue[tinit:,time,:].numel()
        mpiw = torch.mean(yup[:,time,:] -  ylr[:,time,:]).item()
        print(f"picp for the time step {time} is {picp} and mpiw is {mpiw}")
        picplist.append(picp)
        mpiwlist.append(mpiw)

    return picplist, mpiwlist  



def aci_gnn(ytrue, ypred, alpha, gamma, device,correctionmodel, tinit=320, updateMethod="Simple"):
    # 确保ytrue和ypred是PyTorch张量，并转移到指定设备
    ytrue = ytrue.to(device) if not ytrue.is_cuda else ytrue
    ypred = ypred.to(device) if not ypred.is_cuda else ypred
    correctionmodel.to(device)
    correctionmodel.eval()
    T = len(ytrue)  # ytrue: timepoints*12*307*1
    horizon = ytrue.shape[1]
    num_nodes = ytrue.shape[2]

    alphaTrajectory = np.full((T-tinit, horizon,num_nodes), alpha)
    adaptErrSeq = np.zeros((T-tinit, horizon,num_nodes))
    alphat = np.full((horizon,num_nodes), alpha)
    yup = torch.zeros_like(ypred[:T-tinit]).to(device)
    ylr = torch.zeros_like(ypred[:T-tinit]).to(device)
    q = torch.zeros((horizon, num_nodes, 1), device=device)
    with torch.no_grad():  # 避免在推理时保存梯度
        adjustypred = correctionmodel(ypred,ytrue,teacher_forcing_ratio=0)
        
    for t in range(tinit, T):

        YCal, predCal = ytrue[t - tinit:t], ypred[t - tinit:t]  # calibration set
        pred = predCal.squeeze()
        with torch.no_grad():  # 避免在推理时保存梯度
            adjust_pred = correctionmodel(predCal,YCal,teacher_forcing_ratio=0)
        
        adjust_pred = adjust_pred.view(tinit, predCal.shape[1], predCal.shape[2], 1)

        scores = torch.abs(YCal - adjust_pred)  # shape tinit*horizon*num_nodes*1
        #print("the shape of score is{}".format(scores.shape))
        calscores=scores.reshape(tinit,-1)
        alphat = np.clip(alphat, 0, 1)
        #print("the shape of alpha is {}".format(alphat.shape))
        alpha_tensor = torch.tensor(alphat, device=device).unsqueeze(-1)
        #print("the shape of alpha_tensor is{}".format(alpha_tensor.shape))
        alpha_tensor=alpha_tensor.reshape(-1,1).squeeze()
        #a=(torch.ceil((tinit+1)*(1 - alpha_tensor))/tinit).float()
        
        a=(1 - alpha_tensor).float()
        b=torch.ceil((tinit+1)*a)/tinit
        b=torch.clip(b,0,1)
        #print("the shape of b is {}".format(b.shape))
        qu = torch.quantile(calscores, b, dim=0,interpolation='higher')
    
        #print("the shape of adjustypred[t] is {}".format(adjustypred[t].shape))
        #print("the shape of adjustypred[t] is {}".format(adjustypred[t].shape))
        q=torch.diag(qu).reshape(horizon,num_nodes,1)
        #print("the shape of q is {}".format(q.shape))
        # 计算adjusted error sequence
        errSeq = 1 - (q >= torch.abs(adjustypred[t] - ytrue[t])).float()
        #print("the shape of errSeq is {}".format(errSeq.shape))
        #print("the shape ofadaptErrSeq[t - tinit] is {}".format(adaptErrSeq[t - tinit].shape))
        adaptErrSeq[t - tinit] = errSeq.squeeze().cpu().numpy()
        #print("the shape of adaperrSeq is {}".format(adaptErrSeq.shape))
        # 更新alphat
        #print("the shape of alphat is {}".format(alphat.shape))
        if updateMethod == "Simple":
            alphat += gamma * (alpha - adaptErrSeq[t - tinit])
        

        # 计算上下界
       
  
        yup[t - tinit] = adjustypred[t] + q
        ylr[t - tinit] = adjustypred[t] - q

        del YCal, predCal, scores,adjust_pred,calscores  # 释放缓存
        torch.cuda.empty_cache()

    picplist=[]
    mpiwlist=[]
    for time in range (horizon):
        in_num = torch.sum((ytrue[tinit:,time,:] >= ylr[:,time,:]) & (ytrue[tinit:,time,:] <= yup[:,time,:])).item()
        picp = in_num / ytrue[tinit:,time,:].numel()
        mpiw = torch.mean(yup[:,time,:] -  ylr[:,time,:]).item()
        print(f"picp for the time step {time} is {picp} and mpiw is {mpiw}")
        picplist.append(picp)
        mpiwlist.append(mpiw)

    return picplist, mpiwlist  



def torchacicqr(ytrue, ypred,ylow,yup, alpha, gamma, device, confmodel_u,confmodel_l,confmodel_m, tinit=500, updateMethod="Simple"):
    # 确保ytrue和ypred是PyTorch张量，并转移到指定设备
    ytrue = ytrue.to(device) if not ytrue.is_cuda else ytrue
    ypred = ypred.to(device) if not ypred.is_cuda else ypred
    ylow=ylow.to(device) if not ylow.is_cuda else ylow
    yup=yup.to(device) if not yup.is_cuda else yup
    # 将correctionmodel移动到指定设备
    confmodel_m.to(device)
    confmodel_u.to(device)
    confmodel_l.to(device)
    confmodel_m.eval()
    confmodel_l.eval()
    confmodel_u.eval()

    with torch.no_grad():
            adjust_pred = confmodel_m(ypred.squeeze()).unsqueeze(-1)
            adjust_low=confmodel_l(ylow.squeeze()).unsqueeze(-1)
            adjust_up=confmodel_u(yup.squeeze()).unsqueeze(-1)
    T = len(ytrue) # ytrue: timepoints*12*307*1
    horizon = ytrue.shape[1]
    num_nodes = ytrue.shape[2]
    
    alphaTrajectory = np.full((T-tinit, num_nodes, horizon), alpha)
    adaptErrSeq = np.zeros((T-tinit, num_nodes, horizon))
    alphat = np.full((num_nodes, horizon), alpha)
    inupper = torch.zeros_like(ypred[:T-tinit]).to(device)
    inlower = torch.zeros_like(ypred[:T-tinit]).to(device)
    q = torch.zeros((horizon, num_nodes, 1), device=device)

    for t in range(tinit, T):
        YCal, predCal,lowCal,upCal = ytrue[t-tinit:t], ypred[t-tinit:t],ylow[t-tinit:t],yup[t-tinit:t]  # the num of calibration timepoints: tinit
        
        pred=predCal.squeeze()
        low=lowCal.squeeze()
        up=upCal.squeeze()
                    #print("the shape of pred is {}".format(pred.shape))

        with torch.no_grad():
            adjust_pred_cal = confmodel_m(pred)
            adjust_low_cal=confmodel_l(low)

            adjust_up_cal=confmodel_u(up)
                    
                    #print("the shape of adjustpred is {}".format(adjust_pred.shape))
        adjust_pred_cal=adjust_pred_cal.view(tinit,predCal.shape[1],predCal.shape[2],1)
        adjust_low_cal=adjust_low_cal.view(tinit,predCal.shape[1],predCal.shape[2],1)
        adjust_up_cal=adjust_up_cal.view(tinit,predCal.shape[1],predCal.shape[2],1)
        
        scores = torch.maximum(YCal-adjust_up_cal,adjust_low_cal-YCal)
        #print("the shape of adjust[t] is{}".format(adjust_pred[t].shape)) # shape tinit*horizon*num_nodes*1
        for i in range(num_nodes):

            for h in range(horizon):

                if alphat[i, h] <= 0:
                    adaptErrSeq[t-tinit, i, h] = 0
                    alphat[i, h] = 0
                elif alphat[i, h] >= 1:
                    adaptErrSeq[t-tinit, i, h] = 1
                    alphat[i, h] = 1
                else:
                    q[h, i, 0] = torch.quantile(scores[:, h, i, :], 1 - alphat[i, h])

                    #adaptErrSeq[t-tinit, i, h] = 1 - (q[h, i, 0] >= torch.abs((adjust_pred_x[ h, i, 0] - ytrue[t, h, i, 0])/u[h,i,0])).float().item()
                    adaptErrSeq[t-tinit, i, h] = 1 - ((ytrue[t, h, i, 0]>=adjust_low[t,h,i,0]-q[h,i,0]) and (ytrue[t, h, i, 0]<=adjust_up[t,h,i,0]+q[h,i,0])).float().item()
                q[h, i, 0] = torch.quantile(scores[:, h, i, :], 1 - alphat[i, h])
        
        alphaTrajectory[t-tinit] = alphat

        if updateMethod == "Simple":
            alphat += gamma * (alpha - adaptErrSeq[t-tinit])
        #print("the shape of q is{}".format(q.shape))

        inupper[t-tinit] = adjust_up[t] + q
        inlower[t-tinit] = adjust_low[t] - q
        del YCal, predCal,lowCal,upCal,pred, low,up,adjust_pred_cal,adjust_low_cal,adjust_up_cal, scores,   
        torch.cuda.empty_cache() 


    in_num = torch.sum((ytrue[tinit:] >= inlower) & (ytrue[tinit:] <= inupper)).item()
    picp = in_num / ytrue[tinit:].numel()
    mpiw = torch.mean(inupper - inlower).item()
    print(f"picp is {picp} and mpiw is {mpiw}")
    return picp, mpiw

def aci2(ytrue, ypred, alpha, gamma, device, tinit=500, updateMethod="Simple"):
    # 确保ytrue和ypred是PyTorch张量，并转移到指定设备
    ytrue = ytrue.to(device) if not ytrue.is_cuda else ytrue
    ypred = ypred.to(device) if not ypred.is_cuda else ypred

    T = len(ytrue)  # ytrue: timepoints*12*307*1
    horizon = ytrue.shape[1]
    num_nodes = ytrue.shape[2]

    alphaTrajectory = np.full((T - tinit, horizon,num_nodes), alpha)
    adaptErrSeq = np.zeros((T - tinit, horizon, num_nodes))
    alphat = np.full((horizon, num_nodes), alpha)
    yup = torch.zeros_like(ypred[:T - tinit]).to(device)
    ylr = torch.zeros_like(ypred[:T - tinit]).to(device)
    
    for t in range(tinit, T):
        YCal, predCal = ytrue[t - tinit:t], ypred[t - tinit:t]  # calibration set
        scores = torch.abs(YCal - predCal)  # shape tinit*horizon*num_nodes*1
        calscores=scores.reshape(tinit,-1)
        alphat = np.clip(alphat, 0, 1)
        #print("the shape of alpha is {}".format(alphat.shape))
        alpha_tensor = torch.tensor(alphat, device=device).unsqueeze(-1)
        alpha_tensor=alpha_tensor.reshape(-1,1).squeeze()
        # 计算每个时刻、节点和时间步的quantile
        
        #print("the shape of scores is {}".format(calscores.shape))
        #print("the type of alpha is {}".format((1 - alpha_tensor).dtype))
        a=(1 - alpha_tensor).float()
        b=torch.ceil((tinit+1)*a)/tinit
        b=torch.clip(b,0,1)
        #print("the shape of b is {}".format(b.shape))
        qu = torch.quantile(calscores, b, dim=0,interpolation='higher')

        
        q=torch.diag(qu).reshape(-1,horizon,num_nodes,1)
        #print("the shape of q is {}".format(q.shape))
        # 计算adjusted error sequence
        errSeq = 1 - (q >= torch.abs(ypred[t] - ytrue[t])).float()
        #print("the shape of errSeq is {}".format(errSeq.shape))
        adaptErrSeq[t - tinit] = errSeq.squeeze(-1).cpu().numpy()
        #print("the shape of adaperrSeq is {}".format(adaptErrSeq.shape))
        # 更新alphat
        if updateMethod == "Simple":
            alphat += gamma * (alpha - adaptErrSeq[t - tinit])

        # 计算上下界
        yup[t - tinit] = ypred[t] + q
        ylr[t - tinit] = ypred[t] - q

        del YCal, predCal, scores  # 释放缓存
        torch.cuda.empty_cache()

    picplist=[]
    mpiwlist=[]
    for time in range (horizon):
        in_num = torch.sum((ytrue[tinit:,time,:] >= ylr[:,time,:]) & (ytrue[tinit:,time,:] <= yup[:,time,:])).item()
        picp = in_num / ytrue[tinit:,time,:].numel()
        mpiw = torch.mean(yup[:,time,:] -  ylr[:,time,:]).item()
        print(f"picp for the time step {time} is {picp} and mpiw is {mpiw}")
        picplist.append(picp)
        mpiwlist.append(mpiw)

    return picplist, mpiwlist  




def torchaci_correct_s2(ytrue, ypred,x, alpha, gamma, device, correctionmodel, tinit=500, updateMethod="Simple", momentumBW=0.95):
    # 确保ytrue和ypred是PyTorch张量，并转移到指定设备
    ytrue = ytrue.to(device) if not ytrue.is_cuda else ytrue
    ypred = ypred.to(device) if not ypred.is_cuda else ypred
    x=x.to(device) if not ypred.is_cuda else ypred
    # 将correctionmodel移动到指定设备
    correctionmodel.to(device)
    correctionmodel.eval()
    
    T = len(ytrue) # ytrue: timepoints*12*307*1
    horizon = ytrue.shape[1]
    num_nodes = ytrue.shape[2]
    
    alphaTrajectory = np.full((T-tinit, num_nodes, horizon), alpha)
    adaptErrSeq = np.zeros((T-tinit, num_nodes, horizon))
    alphat = np.full((num_nodes, horizon), alpha)
    yup = torch.zeros_like(ypred[:T-tinit]).to(device)
    ylr = torch.zeros_like(ypred[:T-tinit]).to(device)
    q = torch.zeros((horizon, num_nodes, 1), device=device)

    for t in range(tinit, T):
        YCal, predCal,Xcal = ytrue[t-tinit:t], ypred[t-tinit:t],x[t-tinit:t]  # the num of calibration timepoints: tinit
        
        pred = predCal.view(tinit, -1)
        Xcal2=Xcal.view(tinit,-1)
        inputcal=torch.cat((Xcal2,pred),dim=1)#concatenate, on second dimension
        x1=Xcal2[t].view(1,-1)
        predx=ypred[t].view(-1,1)
        inputx=torch.cat((x1,predx),dim=1)


        with torch.no_grad():  # 避免在推理时保存梯度
            #adjust_pred = correctionmodel(inputx)
            adjust_pred = correctionmodel(inputcal)[:,:horizon*num_nodes]
            adjust_u=correctionmodel(inputcal)[:,horizon*num_nodes:]
            u=correctionmodel(inputx)[:,horizon*num_nodes:]
            adjust_pred_x=correctionmodel(inputx)[:,:horizon*num_nodes]
        
        adjust_pred = adjust_pred.view(tinit, predCal.shape[1], predCal.shape[2], 1)
        adjust_u=adjust_u.view(tinit, predCal.shape[1], predCal.shape[2], 1)
        adjust_pred_x = adjust_pred_x.view( predCal.shape[1], predCal.shape[2], 1)
        u = u.view(predCal.shape[1], predCal.shape[2], 1)


        scores = torch.abs((YCal - adjust_pred)/adjust_u)  # shape tinit*horizon*num_nodes*1
        for i in range(num_nodes):
            for h in range(horizon):
                if alphat[i, h] <= 0:
                    adaptErrSeq[t-tinit, i, h] = 0
                    alphat[i, h] = 0
                elif alphat[i, h] >= 1:
                    adaptErrSeq[t-tinit, i, h] = 1
                    alphat[i, h] = 1
                else:
                    q[h, i, 0] = torch.quantile(scores[:, h, i, :], 1 - alphat[i, h])
                    adaptErrSeq[t-tinit, i, h] = 1 - (q[h, i, 0] >= torch.abs((adjust_pred_x[ h, i, 0] - ytrue[t, h, i, 0])/u[h,i,0])).float().item()
                q[h, i, 0] = torch.quantile(scores[:, h, i, :], 1 - alphat[i, h])
        
        alphaTrajectory[t-tinit] = alphat

        if updateMethod == "Simple":
            alphat += gamma * (alpha - adaptErrSeq[t-tinit])
        print(q.shape)
        print(u.shape)
        yup[t-tinit] = ypred[t] + q*u
        ylr[t-tinit] = ypred[t] - q*u

        del YCal, predCal, XCal,pred, adjust_pred,adjust_u,adjust_pred_x,u, scores,   Xcal2,inputcal,x1,predx, inputx
        torch.cuda.empty_cache() 
    picplist=[]
    mpiwlist=[]
    for time in range (horizon):
        in_num = torch.sum((ytrue[tinit:,time,:] >= ylr[:,time,:]) & (ytrue[tinit:,time,:] <= yup[:,time,:])).item()
        picp = in_num / ytrue[tinit:,time,:].numel()
        mpiw = torch.mean(yup[:,time,:] -  ylr[:,time,:]).item()
        print(f"picp for the time step {time} is {picp} and mpiw is {mpiw}")
        picplist.append(picp)
        mpiwlist.append(mpiw)

    return picplist, mpiwlist  