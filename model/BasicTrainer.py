import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
from model.cp.adptivecp import aci,cp,torchacicqr,cp_cqr,aci_cqr,aci2,aci_correction2,aci_gnn
import torch.nn.functional as F
class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output = self.model(data, target, teacher_forcing_ratio=0.)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cuda(), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            #print("size of data is {}".format(data.size()))
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            #data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio)
            #print("size of output is {}".format(output.size()))
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            loss = self.loss(output, label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss
    def train_epoch_cqr(self, epoch):
        self.model.train()
        total_loss = 0
        alpha=0.05
        low_bound=alpha/2
        upp_bound=1-alpha/2
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            #data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio)
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            #print("the shape of output is{}".format(output.shape))
            #print("the shape of label is{}".format(label.shape))
            lower=output[:,:,:,1].unsqueeze(-1)
            upper=output[:,:,:,2].unsqueeze(-1)
            mid=output[:,:,:,0].unsqueeze(-1)
            #print("the shape of lower is{}".format(lower.shape))
            low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
            upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
            #print("the shape of lower_loss is{}".format(low_loss.shape))
            mae_loss= F.l1_loss(mid, label)

            loss = mae_loss + low_loss + upp_loss
            #loss = self.loss(output, label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f},  mse Loss: {:.6f},upper Loss:{:.6f}'.format(epoch, train_epoch_loss, mae_loss,upp_loss))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch_cqr(epoch)
            #print(time.time()-epoch_time)
            #exit()
            '''
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            '''
            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            #val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            '''
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            '''
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            '''
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
            '''
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))



        #self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        #self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, correctionmodel,args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
        #print(args.device)
        model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                output = model(data, target, teacher_forcing_ratio=0)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))

        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        #ytrue=y_true.to_device(args.device)
        #ypred=y_pred.to_device(args.device)
        #np.save('./{}_true.npy'.format(args.dataset), y_true.numpy())
        #np.save('./{}_pred.npy'.format(args.dataset), y_pred.numpy())
        print("start adaptive conformal prediction on the device {}".format(args.device))
        picplist,mpiwlist=aci_gnn(y_pred,y_true,0.05,0.05,args.device,correctionmodel)
        #picplist,mpiwlist=aci2(y_pred,y_true,0.05,0.05,args.device)
        #picplist,mpiwlist=cp(y_pred,y_true,0.05,args.device)

        #picplist,mpiwlist=aci_correction2(y_pred,y_true,0.05,0.05,args.device,correctionmodel)
        #picp,mpiw=torchaci(y_pred,y_true,0.04,0.05,args.device)
        picp=sum(picplist)/len(picplist)
        mpiw=sum(mpiwlist)/len(mpiwlist)
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%,PICP:{:.6f},MPIW:{:.6f}".format(
                    mae, rmse, mape*100,picp,mpiw))
    @staticmethod
    def test_cqr(model, correctionmodel_u,correctionmodel_l,correctionmodel_m,args, data_loader, scaler, path=None):
        #print(args.device)
        model.to(args.device)
        model.eval()
        y_pred = []
        y_up=[]
        y_low=[]
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                output = model(data, target, teacher_forcing_ratio=0)
                lower=output[:,:,:,1].unsqueeze(-1)
                upper=output[:,:,:,2].unsqueeze(-1)
                mid=output[:,:,:,0].unsqueeze(-1)
                
                y_true.append(label)
                y_pred.append(mid)
                y_up.append(upper)
                y_low.append(lower)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))

        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
            y_up = torch.cat(y_up, dim=0)
            y_low = torch.cat(y_low, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            y_up = scaler.inverse_transform(torch.cat(y_up, dim=0))
            y_low = scaler.inverse_transform(torch.cat(y_low, dim=0))
        #ytrue=y_true.to_device(args.device)
        #ypred=y_pred.to_device(args.device)
        #np.save('./{}_true.npy'.format(args.dataset), y_true.numpy())
        #np.save('./{}_pred.npy'.format(args.dataset), y_pred.numpy())
        print("start adaptive conformal prediction on the device")
        
        #picp,mpiw=torchaci(y_pred,y_true,0.05,0.05,args.device,correctionmodel)
        picp,mpiw=aci_cqr(y_pred,y_true,y_low,y_up,0.05,0.05,args.device)

        #picp,mpiw=torchacicqr(y_true,y_pred,y_low,y_up,0.05,0.05,args.device,correctionmodel_u,correctionmodel_l,correctionmodel_m)
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%,PICP:{:.6f},MPIW:{:.6f}".format(
                    mae, rmse, mape*100,picp,mpiw))
        with open("results.txt", "w") as file:  # 打开一个文件用于写入，如果文件不存在则创建它
            for t in range(y_true.shape[1]):
                mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            
                file.write("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%\n".format(t + 1, mae, rmse, mape * 100))

        
            mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        
    
            file.write("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, PICP: {:.6f}, MPIW: {:.6f}\n".format(mae, rmse, mape * 100, picp, mpiw))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))