
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(file_dir)
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.AGCRN import AGCRN as Network
from model.AGCRN2 import AGCRN2 as Network2
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from model.cp.correction import train_cal_correction,ConfMLP,train_cal_correction_gnn,train_cal_correction_CQR
#*************************************************************************#
# Mode = 'train'
# DEBUG = 'True'
# DATASET = 'PEMSD4'      #PEMSD4 or PEMSD8 or PEMS03 or PEMS07
# DEVICE = 'cuda:0'
# MODEL = 'AGCRN'


#parser
initial_parser = argparse.ArgumentParser(description="Parse the dataset name first")
initial_parser.add_argument('--dataset', default='PEMSD8', type=str)
initial_parser.add_argument('--mode', default='train', type=str)
initial_parser.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
initial_parser.add_argument('--debug', default=True, type=bool)
initial_parser.add_argument('--model', default='AGCRN', type=str)
initial_parser.add_argument('--cuda', default=True, type=bool)

initial_args, remaining_argv = initial_parser.parse_known_args()

#get configuration
config_file = './model/{}_{}.conf'.format(initial_args.dataset, initial_args.model)
print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)
print(config.sections())

second_parser = argparse.ArgumentParser(description="Parse additional arguments based on dataset", parents=[initial_parser], add_help=False)
#data
second_parser.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
second_parser.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
second_parser.add_argument('--lag', default=config['data']['lag'], type=int)
second_parser.add_argument('--horizon', default=config['data']['horizon'], type=int)
second_parser.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
second_parser.add_argument('--tod', default=config['data']['tod'], type=eval)
second_parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
second_parser.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
second_parser.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
#model
second_parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
second_parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
second_parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
second_parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
second_parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
second_parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
#train
second_parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
second_parser.add_argument('--seed', default=config['train']['seed'], type=int)
second_parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
second_parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
second_parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
second_parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
second_parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
second_parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
second_parser.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
second_parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
second_parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
second_parser.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
second_parser.add_argument('--teacher_forcing', default=False, type=bool)
#args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
second_parser.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
#test
second_parser.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
second_parser.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
second_parser.add_argument('--log_dir', default='./', type=str)
second_parser.add_argument('--log_step', default=config['log']['log_step'], type=int)
second_parser.add_argument('--plot', default=config['log']['plot'], type=eval)
#save model
second_parser.add_argument('--save_path', default='./model/saved_model/', type=str)
second_parser.add_argument('--save_filename', default='{}_{}.pth'.format(initial_args.dataset, 'saved_model'), type=str)
#calibration
second_parser.add_argument('--correct_epochs', default=100, type=int)
second_parser.add_argument('--correctionmode', default='gnn', type=str)
second_parser.add_argument('--alpha', default=0.05,type=int)
second_parser.add_argument('--size_loss_weight', default=1, type=float)


args = second_parser.parse_args(remaining_argv)

#args = args.parse_args()
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#init model
model = Network(args)
model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
#print_model_parameters(model, only_num=False)

#load dataset
train_loader, cal_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)

from lib.metrics import MAE_torch
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
args.log_dir = log_dir

#start training
trainer = Trainer(model, loss, optimizer, train_loader, cal_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)
print("Current mode: ", args.mode)

if args.mode == 'train':
    trainer.train()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

 # Save the model
    model_save_path = os.path.join(args.save_path, args.save_filename)
    torch.save(model.state_dict(), model_save_path)
    print('***************Model saved successfully at {}'.format(model_save_path))

elif args.mode == 'calcorrection':
    model.load_state_dict(torch.load('./model/saved_model/{}'.format(args.save_filename),map_location=args.device))
    if args.correctionmode=="mlp":
       closs=train_cal_correction(model,args,cal_loader,scaler)
    elif args.correctionmode=='gnn':
       closs=train_cal_correction_gnn(model,args,cal_loader,train_loader,scaler)
    #closs=loss.cpu().numpy()
    loss_cpu = [t.cpu().detach().numpy() for t in closs]
    #print(closs.type())

    plt.plot(loss_cpu)

    plt.savefig('./correction_loss_gnn_plot.png')
elif args.mode == 'test':
    correctmodel=ConfMLP(32*args.num_nodes,args.num_nodes)
    correctmodel.load_state_dict(torch.load('./model/saved_model/correction_t300_{}'.format(args.save_filename),map_location=args.device))
    model.load_state_dict(torch.load('./model/saved_model/{}'.format(args.save_filename),map_location=args.device))
    print("Start testing: Load saved model")
    trainer.test(model, correctmodel,trainer.args, test_loader, scaler, trainer.logger)
elif args.mode == 'testgnn':
    correctmodel=Network(args)
    correctmodel.load_state_dict(torch.load('./model/saved_model/correction_gnn_{}'.format(args.save_filename),map_location=args.device))
    model.load_state_dict(torch.load('./model/saved_model/{}'.format(args.save_filename),map_location=args.device))
    print("Start testing: Load saved model")
    trainer.test(model, correctmodel,trainer.args, test_loader, scaler, trainer.logger)
elif args.mode == 'testcqr':
    correctmodel_u=ConfMLP(10*args.num_nodes,args.num_nodes)
    correctmodel_l=ConfMLP(10*args.num_nodes,args.num_nodes)
    correctmodel_m=ConfMLP(10*args.num_nodes,args.num_nodes)
    correctmodel_u.load_state_dict(torch.load('./model/saved_model/correction_u_{}'.format(args.save_filename),map_location=args.device))
    correctmodel_l.load_state_dict(torch.load('./model/saved_model/correction_l_{}'.format(args.save_filename),map_location=args.device))
    correctmodel_m.load_state_dict(torch.load('./model/saved_model/correction_m_{}'.format(args.save_filename),map_location=args.device))
    model.load_state_dict(torch.load('./model/saved_model/{}'.format(args.save_filename),map_location=args.device))
    model.to(args.device)
    print("Start testing: Load saved model")
    trainer.test_cqr(model, correctmodel_u,correctmodel_l,correctmodel_m,trainer.args, test_loader, scaler)
elif args.mode=="caltest":
    model.load_state_dict(torch.load('./model/saved_model/{}'.format(args.save_filename),map_location=args.device))

    closs=train_cal_correction(model,args,cal_loader,scaler)
    #closs=loss.cpu().numpy()
    loss_cpu = [t.cpu().detach().numpy() for t in closs]
    #print(closs.type())

    plt.plot(loss_cpu)

    plt.savefig('./correction_loss_plot_t400.png')
    #correctmodel=FullyConnectedNetwork(args.num_nodes,2*args.num_nodes,args.horizon)
    correctmodel = ConfMLP(32*args.num_nodes,args.num_nodes).to(args.device)
    correctmodel.load_state_dict(torch.load('./model/saved_model/correction_t400_cross_{}'.format(args.save_filename),map_location=args.device))

    print("Start testing: Load saved model")
    trainer.test(model, correctmodel,trainer.args, test_loader, scaler, trainer.logger)



elif args.mode=="caltestcqr":
    model.load_state_dict(torch.load('./model/saved_model/{}'.format(args.save_filename),map_location=args.device))

    closs=train_cal_correction_CQR(model,args,cal_loader,scaler)
    #closs=loss.cpu().numpy()
    loss_cpu = [t.cpu().detach().numpy() for t in closs]
    #print(closs.type())

    plt.plot(loss_cpu)

    plt.savefig('./correction_loss_plot_cqr.png')
    correctmodel_u=ConfMLP(10*args.num_nodes,args.num_nodes)
    correctmodel_l=ConfMLP(10*args.num_nodes,args.num_nodes)
    correctmodel_m=ConfMLP(10*args.num_nodes,args.num_nodes)
    correctmodel_u.load_state_dict(torch.load('./model/saved_model/correction_u_{}'.format(args.save_filename),map_location=args.device))
    correctmodel_l.load_state_dict(torch.load('./model/saved_model/correction_l_{}'.format(args.save_filename),map_location=args.device))
    correctmodel_m.load_state_dict(torch.load('./model/saved_model/correction_m_{}'.format(args.save_filename),map_location=args.device))
    model.load_state_dict(torch.load('./model/saved_model/{}'.format(args.save_filename),map_location=args.device))
    model.to(args.device)
    print("Start testing: Load saved model")
    trainer.test_cqr(model, correctmodel_u,correctmodel_l,correctmodel_m,trainer.args, test_loader, scaler)