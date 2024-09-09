import numpy as np
from scipy.spatial import distance
#from statsmodels.regression.quantile_regression import QuantReg
#from arch import arch_model
import torch
import matplotlib.pyplot as plt
def OfflineCI(ytrue, ypred, alpha, gamma, tinit=500, updateMethod="Simple", momentumBW=0.95):# CI with  previous $tinit$ as calibration set. offline training! 
    T = len(ytrue)
    alphaTrajectory = np.full((T-tinit,ytrue.shape[2]), alpha)
    adaptErrSeq = np.zeros((T-tinit,ytrue.shape[2]))#shape: number of test timepoints*number of nodes
    alphat = np.full(ytrue.shape[2],alpha)# shape: 307nodes
    yup=np.zeros_like(ypred)[:T-tinit,]
    
    ylr=np.zeros_like(ypred)[:T-tinit,]

    for t in range(tinit, T):

        YCal,predCal = ytrue[t-tinit:t],ypred[t-tinit:t]

        scores = np.abs(YCal-predCal) #500*307*12
        plt.plot(scores[:,0,0,:])
        

 
        

        adaptErrSeq[t-tinit,alphat<=0] = 0
        for i in range (ytrue.shape[2]):
            if i==0:
               if alphat[i]<=0:
                  print("alphat hit 0!!")
                  adaptErrSeq[t-tinit,i] = 0
                  alphat[i]=0
               q = np.quantile(scores[:,:,i,:], 1-alphat[i],axis=0)
               adaptErrSeq[t-tinit,i] = np.any (q<np.abs(ypred[t,:,i]-ytrue[t,:,i]))
            if i>0:
                if alphat[i]>=1:
                  print("alphat hit 1!")
                  adaptErrSeq[t-tinit,i] = 1
                  alphat[i]=1
                  q =np.concatenate((q,np.quantile(scores[:,:,i,:], 1-alphat[i],axis=0)),axis=1)
                if alphat[i]<=0:
                  print("alphat hit 0! during time {} for node{}".format(t,i))

                  adaptErrSeq[t-tinit,i] = 0
                  alphat[i]=0
                  q =np.concatenate((q,np.quantile(scores[:,:,i,:], 1-alphat[i],axis=0)),axis=1)
                if alphat[i]>0 and alphat[i]<1:
                  tempq=np.quantile(scores[:,:,i,:], 1-alphat[i],axis=0)# shape 12*1
                  q =np.concatenate((q,np.quantile(scores[:,:,i,:], 1-alphat[i],axis=0)),axis=1) # q shape 12*307
                  adaptErrSeq[t-tinit,i] = 1-np.all (tempq>=np.abs(ypred[t,:,i]-ytrue[t,:,i]))


        q=q.reshape(q.shape[0],-1,1)# shape 12*307*1

    
        alphaTrajectory[t-tinit] = alphat 

        if updateMethod == "Simple":
            alphat += gamma * (alpha - adaptErrSeq[t-tinit])

        yup[t-tinit]=ypred[t]+q

        ylr[t-tinit]=ypred[t]-q




        

    in_num = np.sum((ytrue[tinit:,] >= ylr)&(ytrue[tinit:,] <= yup ))
    
    picp = in_num/(ytrue[tinit:,].shape[0]*ytrue[tinit:,].shape[1]*ytrue[tinit:,].shape[2])

    mpiw = np.mean(yup-ylr)

    return picp,mpiw
'''


        
        elif updateMethod == "Momentum":
            w = momentumBW ** np.arange(1, t-tinit+2)[::-1]
            w /= w.sum()
            alphat += gamma * (alpha - np.dot(adaptErrSeq[:t-tinit+1], w))

        if t % 100 == 0:
            print(f"Done {t} time steps")
'''
def OfflineCItorch(ytrue, ypred, alpha, gamma, tinit=500, updateMethod="Simple", momentumBW=0.95):
    T = len(ytrue)
    alphaTrajectory = torch.full((T - tinit, ytrue.shape[2]), alpha, device=ytrue.device)
    adaptErrSeq = torch.zeros((T - tinit, ytrue.shape[2]), device=ytrue.device)
    alphat = torch.full((ytrue.shape[2],), alpha, device=ytrue.device)
    yup = torch.zeros_like(ypred)[:T - tinit]
    ylr = torch.zeros_like(ypred)[:T - tinit]

    for t in range(tinit, T):
        YCal, predCal = ytrue[t - tinit:t], ypred[t - tinit:t]
        scores = torch.abs(YCal - predCal)

        adaptErrSeq[t - tinit, alphat <= 0] = 0
        for i in range(ytrue.shape[2]):
            if i == 0:
                if alphat[i] <= 0:
                    #print("alphat hit 0!!")
                    adaptErrSeq[t - tinit, i] = 0
                    alphat[i] = 0
                q = torch.quantile(scores[:, :, i, :], 1 - alphat[i], dim=0)
                adaptErrSeq[t - tinit, i] = torch.any(q < torch.abs(ypred[t, :, i] - ytrue[t, :, i]))
            if i > 0:
                if alphat[i] >= 1:
                    #print("alphat hit 1!")
                    adaptErrSeq[t - tinit, i] = 1
                    alphat[i] = 1
                    q = torch.cat((q, torch.quantile(scores[:, :, i, :], 1 - alphat[i], dim=0)), dim=1)
                if alphat[i] <= 0:
                    #print("alphat hit 0! during time {} for node{}".format(t, i))
                    adaptErrSeq[t - tinit, i] = 0
                    alphat[i] = 0
                    q = torch.cat((q, torch.quantile(scores[:, :, i, :], 1 - alphat[i], dim=0)), dim=1)
                if alphat[i] > 0 and alphat[i] < 1:
                    tempq = torch.quantile(scores[:, :, i, :], 1 - alphat[i], dim=0)
                    q = torch.cat((q, torch.quantile(scores[:, :, i, :], 1 - alphat[i], dim=0)), dim=1)
                    adaptErrSeq[t - tinit, i] = torch.any(tempq < torch.abs(ypred[t, :, i] - ytrue[t, :, i]))

        q = q.view(q.shape[0], -1, 1)

        alphaTrajectory[t - tinit] = alphat

        if updateMethod == "Simple":
            alphat += gamma * (alpha - adaptErrSeq[t - tinit])

        yup[t - tinit] = ypred[t] + q
        ylr[t - tinit] = ypred[t] - q

    in_num = torch.sum((ytrue[tinit:] >= ylr) & (ytrue[tinit:] <= yup))
    picp = in_num.item() / (ytrue[tinit:].numel())
    mpiw = torch.mean(yup - ylr).item()

    return picp, mpiw

ypred=np.load("PEMSD4_pred.npy")
ytrue=np.load("PEMSD4_true.npy")

