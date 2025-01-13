
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
from lib.TrainInits import print_model_parameters
from model.AGCRN import AGCRN as Network

class ConfMLP(torch.nn.Module):
    def __init__(self, confnn_hidden_dim, output_dim):
        super().__init__()
        self.conf1 =nn.Linear(output_dim,  confnn_hidden_dim)  
        self.conf2=nn.Linear(confnn_hidden_dim,confnn_hidden_dim)
        self.conf3=nn.Linear(confnn_hidden_dim,output_dim)
    def forward(self, x):#x is score, in our case, x=abs(pred-true)
        with torch.no_grad():
             scores = x
        adjust_scores = torch.relu(self.conf1(scores))
        adjust_scores = torch.relu(self.conf2(adjust_scores))
        adjust_scores = self.conf3(adjust_scores)
        return adjust_scores
def fit_calibration(temp_model, eval, data, train_mask, test_mask, patience = 100):
    """
    Train calibrator
    """    
    vlss_mn = float('Inf')
    with torch.no_grad():
        logits = temp_model.model(data.x, data.edge_index)
        labels = data.y
        edge_index = data.edge_index
        model_dict = temp_model.state_dict()
        parameters = {k: v for k,v in model_dict.items() if k.split(".")[0] != "model"}
    for epoch in range(2000):
        temp_model.optimizer.zero_grad()
        temp_model.train()
        # Post-hoc calibration set the classifier to the evaluation mode
        temp_model.model.eval()
        assert not temp_model.model.training
        calibrated = eval(logits)
        loss = F.cross_entropy(calibrated[train_mask], labels[train_mask])
        # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
        # margin_reg = 0.
        # loss = loss + margin_reg * dist_reg
        loss.backward()
        temp_model.optimizer.step()

        with torch.no_grad():
            temp_model.eval()
            calibrated = eval(logits)
            val_loss = F.cross_entropy(calibrated[test_mask], labels[test_mask])
            # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
            # val_loss = val_loss + margin_reg * dist_reg
            if val_loss <= vlss_mn:
                state_dict_early_model = copy.deepcopy(parameters)
                vlss_mn = np.min((val_loss.cpu().numpy(), vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
    model_dict.update(state_dict_early_model)
    temp_model.load_state_dict(model_dict)


def train_cal_correction(model,args,data_loader,scaler,conf_correct_model='mlp',tinit=300):
            model_to_correct = copy.deepcopy(model)

            if conf_correct_model == 'mlp':
                confmodel = ConfMLP(32*args.num_nodes,args.num_nodes).to(args.device)
            
            optimizer = torch.optim.Adam(confmodel.parameters(), weight_decay=5e-4, lr=0.0001)  
            pred_loss_hist, size_loss_hist, cons_loss_hist, val_size_loss_hist = [], [], [], []
            best_size_loss = 10000
            best_val_acc = 0
            y_true=[]
            y_pred=[]
            alpha=args.alpha
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(data_loader):
                     data = data[..., :args.input_dim]
                     #print("data size is {}".format(data.size()))
                     label = target[..., :args.output_dim]
                     output =model_to_correct(data, target, teacher_forcing_ratio=0)

                     y_true.append(label)
                     y_pred.append(output)

                y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
                if args.real_value:
                   y_pred = torch.cat(y_pred, dim=0)
                else:
                   y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            print('Starting topology-aware conformal correction...')
            for epoch in range(1, args.correct_epochs + 1):  
                T = len(y_true)
                rloss=[]
                predtest=int(T/3)
                for t in range(tinit+predtest, T):
                    YCal, predCal = y_true[t-tinit:t], y_pred[t-tinit:t]#the num of calibration timepoints: tinit
                    #scores = torch.abs(YCal - predCal)#shape tinit*horizon*num_nodes*1

                    sample_indices = np.random.choice(np.arange(predtest), size=300, replace=False)
                    Ytest, Yhattest = y_true[sample_indices], y_pred[sample_indices] 

                    confmodel.train()
                    
                    optimizer.zero_grad()
                    adjust_pred_cal = confmodel(predCal.squeeze())
                    
                    #print("the shape of adjustpred is {}".format(adjust_pred.shape))
                    adjust_pred_cal=adjust_pred_cal.view(tinit,predCal.shape[1],predCal.shape[2],1)
                    #print("the shape of adjustpred is {}".format(adjust_pred.shape))
                    adjust_pred=confmodel(Yhattest.squeeze())
                    #print(adjust_pred.shape)
                    #print(YCal.shape)
                    scores = torch.abs(YCal - adjust_pred_cal)
                    qhat = torch.quantile(scores, np.ceil((tinit+1)*(1-alpha))/tinit, interpolation='higher',dim=0)
                    # Get the score quantile                                                
                    size_loss = torch.mean(qhat)
                    pred_loss=torch.mean(torch.abs(Ytest-adjust_pred.unsqueeze(-1)))
                    if epoch<args.correct_epochs/2:
                        loss=pred_loss

                    elif epoch>=args.correct_epochs/2:
                        loss = pred_loss+args.size_loss_weight* size_loss
                    rloss.append(loss)
                    loss.backward()
                    optimizer.step()

                print('**********Correction Epoch {}: Loss (average of qhat): {:.6f}'.format(epoch, loss))
            
            torch.save(confmodel.state_dict(), '{}correction_mlp_{}_{}'.format(args.save_path,tinit,args.save_filename))
            print("correction model saved!")

            return rloss

def get_ecu_graph_distance(x,y):
    # Squeeze the last dimension to get rid of the trailing 1
    x_squeezed = x.squeeze(-1)  # Shape: [300, 12, 50]
    y_squeezed = y.squeeze(-1)  # Shape: [300, 12, 50]

    # Compute the squared difference
    squared_diff = (x_squeezed - y_squeezed) ** 2  # Shape: [300, 12, 50]

    # Sum the squared differences over the node dimension (dimension 2, i.e., size 50)
    sum_squared_diff = squared_diff.sum(dim=-1)  # Shape: [300, 12]

    # Compute the Euclidean distance by taking the square root
    euclidean_distance = torch.sqrt(sum_squared_diff)  # Shape: [300, 12]

    # Reshape or view the result to get the desired output shape [300, 12, 1]
    output = euclidean_distance.unsqueeze(-1)  # Shape: [300, 12, 1]
    return output

def graph_size_loss(y_pred,y_true,confmodel,tinit,alpha,size_loss_weight,rloss):
    T = len(y_true)
    predtest = int(T / 3)
    optimizer = torch.optim.Adam(confmodel.parameters(), weight_decay=5e-4, lr=0.001)


    for t in range(tinit + predtest, T):
        # Prediction Accuracy Loss
        sample_indices = np.random.choice(np.arange(predtest), size=100, replace=False)
        Ytest, Yhattest = y_true[sample_indices], y_pred[sample_indices]
        adjust_pred = confmodel(Yhattest, Ytest, teacher_forcing_ratio=0)
        pred_loss = torch.mean(torch.abs(Ytest - adjust_pred))

        # CP Loss
        YCal, predCal = y_pred[t - tinit:t], y_true[t - tinit:t]
        adjust_pred_cal = confmodel(predCal, YCal, teacher_forcing_ratio=0)
        adjust_pred_cal = adjust_pred_cal.view(tinit, predCal.shape[1], predCal.shape[2], 1)
        #  scores: torch.Size([300, 12, 50, 1])
        #scores = torch.abs(YCal - adjust_pred_cal)
        scores = get_ecu_graph_distance( YCal , adjust_pred_cal )
        qhat = torch.quantile(scores, np.ceil((tinit + 1) * (1 - alpha)) / tinit, interpolation='higher', dim=0)
        # qhat: torch.Size([12, 50, 1])
        size_loss = torch.mean(qhat)
        # print(qhat.shape)

        # Train
        confmodel.train()
        optimizer.zero_grad()

        loss = pred_loss + size_loss_weight * size_loss
        rloss.append(loss)
        loss.backward()
        optimizer.step()

        del YCal, predCal, sample_indices, Ytest, Yhattest, adjust_pred_cal, adjust_pred, scores, qhat, size_loss, pred_loss
        torch.cuda.empty_cache()

    return confmodel, rloss



def single_node_size_loss(y_pred,y_true,confmodel,tinit,alpha,size_loss_weight,rloss):
    T = len(y_true)
    predtest = int(T / 3)
    optimizer = torch.optim.Adam(confmodel.parameters(), weight_decay=5e-4, lr=0.001)

    for t in range(tinit + predtest, T):
        YCal, predCal = y_pred[t - tinit:t], y_true[t - tinit:t]  # the num of calibration timepoints: tinit
        # scores = torch.abs(YCal - predCal)#shape tinit*horizon*num_nodes*1

        sample_indices = np.random.choice(np.arange(predtest), size=100, replace=False)

        Ytest, Yhattest = y_true[sample_indices], y_pred[sample_indices]

        # print("the shape of pred is {}".format(predCal.shape))
        confmodel.train()

        optimizer.zero_grad()

        adjust_pred_cal = confmodel(predCal, YCal, teacher_forcing_ratio=0)

        # print("the shape of adjustpred is {}".format(adjust_pred_cal.shape))
        adjust_pred_cal = adjust_pred_cal.view(tinit, predCal.shape[1], predCal.shape[2], 1)
        # print("the shape of adjustpred is {}".format(adjust_pred.shape))
        adjust_pred = confmodel(Yhattest, Ytest, teacher_forcing_ratio=0)
        # print(adjust_pred.shape)

        scores = torch.abs(YCal - adjust_pred_cal)

        qhat = torch.quantile(scores, np.ceil((tinit + 1) * (1 - alpha)) / tinit, interpolation='higher', dim=0)
        # Get the score quantile
        size_loss = torch.mean(qhat)
        pred_loss = torch.mean(torch.abs(Ytest - adjust_pred))

        loss = pred_loss + size_loss_weight * size_loss
        rloss.append(loss)
        loss.backward()
        optimizer.step()
        del YCal, predCal, sample_indices, Ytest, Yhattest, adjust_pred_cal, adjust_pred, scores, qhat, size_loss, pred_loss
        torch.cuda.empty_cache()

    return confmodel, rloss



def train_cal_correction_gnn(model,args,cal_loader,train_loader,scaler,tinit=300):
            model_to_correct = copy.deepcopy(model)

            confmodel =Network(args).to(args.device)
            print("use GNN for correction")
            for p in confmodel.parameters():
                    if p.dim() > 1:
                       nn.init.xavier_uniform_(p)
                    else:
                       nn.init.uniform_(p)
            print_model_parameters(confmodel, only_num=False)
            optimizer = torch.optim.Adam(confmodel.parameters(), weight_decay=5e-4, lr=0.001)  
            pred_loss_hist, size_loss_hist, cons_loss_hist, val_size_loss_hist = [], [], [], []
            best_size_loss = 10000
            y_true=[]
            y_pred=[]
            alpha=args.alpha

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(cal_loader):
                     data = data[..., :args.input_dim]
                     #print("data size is {}".format(data.size()))
                     label = target[..., :args.output_dim]
                     output =model_to_correct(data, target, teacher_forcing_ratio=0)

                     y_true.append(label)
                     y_pred.append(output)

                y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
                if args.real_value:
                       y_pred = torch.cat(y_pred, dim=0)

            for epoch in range(1, args.correct_epochs + 1):  
                T = len(y_true)
                rloss=[]
                if epoch<=int(args.correct_epochs*3/4):
                   for batch_idx, (data, target) in enumerate(train_loader):
                     data = data[..., :args.input_dim]
                     #print("data size is {}".format(data.size()))
                     label = target[..., :args.output_dim]
                     with torch.no_grad():
                         output =model_to_correct(data, label, teacher_forcing_ratio=0)
                         #print("the shape of the input of confmodel is {}".format(output.shape))
                     confmodel.train()
                     optimizer.zero_grad()
                     label = scaler.inverse_transform(label)
                     adjust_pred=confmodel(label, output, teacher_forcing_ratio=0)
                     if torch.isnan(output).any() or torch.isnan(label).any():
                              print("NaN detected in input or labels")
                     if torch.isnan(adjust_pred).any():
                              print("NAN detected in adjust_predictions or labels")
                     #print("the shape of the output of confmodel is {}".format(adjust_pred.shape))
                    
                     maeloss= nn.L1Loss().to(args.device)
                     loss = maeloss(adjust_pred, output)
                     rloss.append(loss)
                     loss.backward()
                     optimizer.step()
                else:
                    confmodel, rloss = graph_size_loss(y_pred, y_true, confmodel, tinit, alpha, args.size_loss_weight,rloss)
                print('**********Correction Epoch {}: Loss (prediction loss): {:.6f}'.format(epoch, rloss[-1]))


            torch.save(confmodel.state_dict(), '{}correction_gnn_{}_{}'.format(args.save_path,tinit,args.save_filename))
            print("correction model saved!")

            return rloss

# def train_cal_correction_CQR(model,args,data_loader,scaler,conf_correct_model='mlp',tinit=300):
#             model_to_correct = copy.deepcopy(model)
#             if conf_correct_model == 'gnn':
#                 #confmodel = ConfGNN(32*args.num_nodes, output_dim=args.horizon*args.num_nodes).to(args.device)
#                 print("use GNN for correction")
#             elif conf_correct_model == 'mlp':
#                 confmodel_m = ConfMLP(10*args.num_nodes,args.num_nodes).to(args.device)
#                 confmodel_u = ConfMLP(10*args.num_nodes,args.num_nodes).to(args.device)
#                 confmodel_l = ConfMLP(10*args.num_nodes,args.num_nodes).to(args.device)
#             alpha=0.05
#             low_bound=alpha/2
#             upp_bound=1-alpha/2
#             optimizer = optimizer = torch.optim.Adam(list(confmodel_m.parameters()) + list(confmodel_u.parameters()) + list(confmodel_l.parameters()),weight_decay=5e-4, lr=0.001)
#             pred_loss_hist, size_loss_hist, cons_loss_hist, val_size_loss_hist = [], [], [], []
#             best_size_loss = 10000
#             best_val_acc = 0
#             y_true=[]
#             y_pred=[]
#             y_low=[]
#             y_up=[]
#             alpha=args.alpha
#             with torch.no_grad():
#                 for batch_idx, (data, target) in enumerate(data_loader):
#                      data = data[..., :args.input_dim]
#                      #print("data size is {}".format(data.size()))
#                      label = target[..., :args.output_dim]
#                      output =model_to_correct(data, target, teacher_forcing_ratio=0)
#                      mid=output[:,:,:,0].unsqueeze(-1)
#                      upper=output[:,:,:,2].unsqueeze(-1)
#                      lower=output[:,:,:,1].unsqueeze(-1)
#                      y_true.append(label)
#                      y_pred.append(mid)
#                      y_low.append(lower)
#                      y_up.append(upper)
#
#                 y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
#                 if args.real_value:
#                    y_pred = torch.cat(y_pred, dim=0)
#                    y_low = torch.cat(y_low, dim=0)
#                    y_up = torch.cat(y_up, dim=0)
#                 else:
#                    y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
#                    y_low = scaler.inverse_transform(torch.cat(y_low, dim=0))
#                    y_up = scaler.inverse_transform(torch.cat(y_up, dim=0))
#             print('Starting topology-aware conformal correction...')
#             for epoch in range(1, args.correct_epochs + 1):
#                 T = len(y_true)
#                 Cal_length=int(T/2)
#                 Test_length=Cal_length+1
#                 rloss=[]
#                 confmodel_m.train()
#                 confmodel_l.train()
#                 confmodel_u.train()
#                 for t in range(tinit, Cal_length):
#                     YCal, predCal,low,up = y_true[t-tinit:t], y_pred[t-tinit:t],y_low[t-tinit:t],y_up[t-tinit:t]#the num of calibration timepoints: tinit
#                     #scores = torch.abs(YCal - predCal)#shape tinit*horizon*num_nodes*1
#
#                     pred=predCal.squeeze()
#                     low=low.squeeze()
#                     up=up.squeeze()
#                     yTrue=YCal.squeeze()
#                     #print("the shape of pred is {}".format(pred.shape))
#
#                     optimizer.zero_grad()
#                     adjust_pred = confmodel_m(pred)
#                     adjust_low=confmodel_l(low)
#                     adjust_up=confmodel_u(up)
#
#                     #print("the shape of adjustpred is {}".format(adjust_pred.shape))
#                     adjust_pred=adjust_pred.view(tinit,predCal.shape[1],predCal.shape[2],1)
#                     adjust_low=adjust_low.view(tinit,predCal.shape[1],predCal.shape[2],1)
#                     adjust_up=adjust_up.view(tinit,predCal.shape[1],predCal.shape[2],1)
#                     yTrue=yTrue.view(tinit,predCal.shape[1],predCal.shape[2],1)
#                     #print(torch.sum(adjust_low<=yTrue)/(yTrue.shape[0]*yTrue.shape[1]*yTrue.shape[2]))
#                     cal_scores = torch.maximum(yTrue-adjust_up,adjust_low-yTrue)
#                     # Get the score quantile
#                     qhat = torch.quantile(cal_scores, np.ceil((tinit+1)*(1-alpha))/tinit, interpolation='higher',dim=0)
#                     y_truetest,predtest,lowtest,uptest= y_true[Cal_length+t], y_pred[Cal_length+t],y_low[Cal_length+t],y_up[Cal_length+t]
#                     predtest=predtest.squeeze()
#                     lowtest=lowtest.squeeze()
#                     uptest=uptest.squeeze()
#                     adjust_predtest = confmodel_m(predtest)
#                     adjust_lowtest=confmodel_l(low)
#                     adjust_uptest=confmodel_u(up)
#                     qhat=qhat.squeeze()
#                     #print(torch.mean(qhat))
#                     ylowt=y_low[t].squeeze()
#                     yupt=y_up[t].squeeze()
#                     size_loss = torch.mean(yupt + qhat - (ylowt - qhat))
#                     #print(size_loss)
#                     #size_loss_hist.append(size_loss)
#
#                     low_loss = torch.mean(torch.max((low_bound - 1) * (y_truetest.squeeze() - adjust_lowtest), low_bound * (y_truetest.squeeze() - adjust_lowtest)))
#                     upp_loss = torch.mean(torch.max((upp_bound - 1) * (y_truetest.squeeze() - adjust_uptest), upp_bound * (y_truetest.squeeze() - adjust_uptest)))
#
#                     mae_loss= F.l1_loss(y_truetest.squeeze(), adjust_predtest)
#
#                     pred_loss = mae_loss + low_loss + upp_loss
#                     #print("the value of predict_loss is{}".format(pred_loss))
#                     #pred_loss_hist.append(pred_loss)
#                     if epoch<args.correct_epochs/2:
#                         loss=pred_loss
#
#                     elif epoch>=args.correct_epochs/2:
#                         loss = pred_loss+args.size_loss_weight* size_loss
#                     rloss.append(loss)
#                     loss.backward()
#                     optimizer.step()
#                     #print('step {}: Loss: {:.6f}'.format(t-tinit, loss))
#                     del YCal, predCal,low,up,y_truetest,predtest,lowtest,uptest,adjust_low,adjust_lowtest,adjust_pred,adjust_predtest,adjust_up,adjust_uptest,qhat,low_loss,upp_loss,mae_loss,size_loss
#                     torch.cuda.empty_cache()
#                 print('**********Correction Epoch {}: Loss: {:.6f}'.format(epoch, loss))
#
#             torch.save(confmodel_m.state_dict(), './saved_model/correction_m_{}'.format(args.save_filename))
#             torch.save(confmodel_u.state_dict(), './saved_model/correction_u_{}'.format(args.save_filename))
#             torch.save(confmodel_l.state_dict(), './saved_model/correction_l_{}'.format(args.save_filename))
#             print("correction model saved!")
#
#             return rloss
#
# def train_cal_correction_S2(model,args,data_loader,scaler,conf_correct_model='mlp',tinit=500):
#             model_to_correct = copy.deepcopy(model)
#             if conf_correct_model == 'gnn':
#                 print("use GNN for correction")
#                 #confmodel = ConfGNN(32*args.num_nodes, output_dim=args.horizon*args.num_nodes).to(args.device)
#
#             elif conf_correct_model == 'mlp':
#                 confmodel1 = ConfMLP(32*args.num_nodes,args.horizon*args.num_nodes*2).to(args.device)
#                 confmodel = ConfMLP(32*args.num_nodes,args.horizon*args.num_nodes*2).to(args.device)
#             params = list(confmodel.parameters()) + list(confmodel1.parameters())
#             optimizer = torch.optim.Adam(params, weight_decay=5e-4, lr=0.001)
#             optimizer1 = torch.optim.Adam(confmodel.parameters(), weight_decay=5e-4, lr=0.001)
#             optimizer2 = torch.optim.Adam(confmodel1.parameters(), weight_decay=5e-4, lr=0.001)
#             pred_loss_hist, size_loss_hist, cons_loss_hist, val_size_loss_hist = [], [], [], []
#             best_size_loss = 10000
#             best_val_acc = 0
#             y_true=[]
#             y_pred=[]
#             x=[]
#             rloss=[]
#             alpha=args.alpha
#             with torch.no_grad():
#                 for batch_idx, (data, target) in enumerate(data_loader):
#                      data = data[..., :args.input_dim]
#                      #print("data size is {}".format(data.size()))
#                      label = target[..., :args.output_dim]
#                      output =model_to_correct(data, target, teacher_forcing_ratio=0)
#                      x.append(data)
#                      y_true.append(label)
#                      y_pred.append(output)
#                 x=scaler.inverse_transform(torch.cat(x, dim=0))
#                 #print(x.shape) #time_points*num_nodes*12*1
#                 y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
#                 if args.real_value:
#                    y_pred = torch.cat(y_pred, dim=0)
#                 else:
#                    y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
#             print('Starting topology-aware conformal correction...')
#             mode_collapse=False
#             for epoch in range(1, args.correct_epochs + 1):
#                 T = len(y_true)
#                 for t in range(tinit, T):
#                     YCal, predCal,x1 = y_true[t-tinit:t], y_pred[t-tinit:t],x[t-tinit:t]#the num of calibration timepoints: tinit
#                     #scores = torch.abs(YCal - predCal)#shape tinit*horizon*num_nodes*1
#                     pred=predCal.view(tinit,-1)
#                     #print(x1.shape)
#                     x2=x1.view(tinit,-1)
#                     inputx=torch.cat((x2,pred),dim=1)#concatenate, on second dimension
#                     confmodel.train()
#                     confmodel1.train()
#                     optimizer.zero_grad()
#                     adjust_pred = confmodel1(inputx)[:,:args.horizon*args.num_nodes]
#                     adjust_u=confmodel(inputx)[:,args.horizon*args.num_nodes:]
#
#                     adjust_pred=adjust_pred.view(tinit,predCal.shape[1],predCal.shape[2],1)
#                     adjust_u=adjust_u.view(tinit,predCal.shape[1],predCal.shape[2],1)
#
#                     scores = torch.abs((YCal - adjust_pred)/adjust_u)
#                     qhat = torch.quantile(scores, np.ceil((tinit+1)*(1-alpha))/tinit, interpolation='higher',dim=0)
#
#                     size_loss = torch.mean(torch.abs(qhat*adjust_u))
#                     pred_loss=torch.mean(torch.abs(adjust_pred- YCal))
#                     if epoch<10:
#                         loss=pred_loss
#                     elif epoch>=10:
#                         loss = pred_loss+args.size_loss_weight* size_loss
#                     if torch.isnan(loss):
#                         print("the loss is nan, model collapse!")
#                         print('The size_loss: {:.6f}, the max of qhat: {:.6f},the min of qhat:{},the max of adjust_u: {:.6f},the min of adjust_u:{}'.format(size_loss, torch.max(qhat), torch.min(qhat),torch.min(adjust_u),torch.min(adjust_u)))
#                         mode_collapse=True
#                         break
#                     rloss.append(loss)
#                     loss.backward()
#
#                     optimizer.step()
#                     max_weight = max(param.abs().max().item() for param in confmodel.parameters())
#                     max_weight1 = max(param.abs().max().item() for param in confmodel1.parameters())
#
#
#                 print('Correction Epoch {}: Loss (average of qhat): {:.6f}, max of adjust scores:{:.6f},max of adjusted u:{:.6f}'.format(epoch, loss,max_weight1,max_weight))
#                 if mode_collapse== True:
#                     break
#
#             torch.save(confmodel.state_dict(), './model/saved_model/correction_{}_s2'.format(args.save_filename))
#             torch.save(confmodel1.state_dict(), './model/saved_model/correction_{}_s21'.format(args.save_filename))
#             print("correction model saved!")
#
#             return rloss
'''
            
model = Network(args)
model = model.to(args.device)

model.load_state_dict(torch.load('./model/saved_model/{}'.format(args.save_filename),map_location=args.device))

#load dataset
train_loader, cal_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)
                    
train_cal_correction(model,args,cal_loader,scaler)
'''