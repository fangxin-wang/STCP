import os
import sys
import pickle
from torch_geometric.utils import from_networkx

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(file_dir)
import matplotlib.pyplot as plt
import torch
import re
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.AGCRN import AGCRN, InferenceWithDropout, GT_Predictor
# from model.PCP import PCP
from model.correction import RKHSMapping
from model.BasicTrainer import Trainer, convert_str_2_tensor
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from correction import train_cal_correction,train_cal_correction_gnn
from STPredictor import RecurrentGCN, TemporalGNN
# Change the import to use our own DCRNN implementation
from model.DCRNN import DCRNN
import networkx as nx

torch.set_default_dtype(torch.float32)
#*************************************************************************#


#parser
initial_parser = argparse.ArgumentParser(description="Parse the dataset name first")
initial_parser.add_argument('--dataset', default='PEMSD8', type=str)
initial_parser.add_argument('--syn_seed', default=1007, type=int)
initial_parser.add_argument('--mode', default='train', type=str)
initial_parser.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
initial_parser.add_argument('--debug', default=True, type=bool)
initial_parser.add_argument('--model', default='AGCRN', type=str)
initial_parser.add_argument('--cuda', default=True, type=bool)
initial_parser.add_argument('--tinit',default=300,type=int)
initial_parser.add_argument('--K',default=1,type=int)
initial_parser.add_argument("--ACI_MLP_test", action="store_true", default=False, help="Test with CP")
initial_parser.add_argument("--PCP_test", action="store_true", default=False, help="Test with PCP")
initial_parser.add_argument("--PCP_ellip_test", action="store_true", default=False, help="Test with PCP_ellip")
initial_parser.add_argument("--ACI_test", action="store_true", default=False, help="Test with ACI")
initial_parser.add_argument('--gamma',default=0.05,type=float, help = 'Gamma for ACI Step')
initial_parser.add_argument("--link_pred", action="store_true", default=False, help="Add link prediction loss to score")

#initial_parser.add_argument("--ACI_GNN_test", action="store_true", default=False, help="Test with ACI_GNN")
#initial_parser.add_argument("--CP_MLP_test", action="store_true", default=False, help="Test with CP MLP")
#initial_parser.add_argument("--ACI_MLP_test", action="store_true", default=False, help="Test with ACI MLP")

initial_args, remaining_argv = initial_parser.parse_known_args()


def parse_d_from_str(s: str) -> int:
    # Check for "top_N" format
    match = re.match(r"PEMS03_top_(\d+)", s)
    if match:
        return int(match.group(1))
    # Check for "w12" format
    elif "PEMS03_w12" in s:
        return "w12"
    else:
        return False


#get configuration
print(initial_args.dataset, parse_d_from_str(initial_args.dataset) )
if initial_args.dataset=='syn_gpvar':
    config_file = './model/syn_gpvar_{}.conf'.format( initial_args.syn_seed )
elif initial_args.dataset=='syn_tailup':
    config_file = './model/syn_tailup_{}.conf'.format( initial_args.syn_seed )
elif parse_d_from_str(initial_args.dataset) == "w12":
    config_file = './model/PEMS03_w12_{}.conf'.format(initial_args.model)
elif parse_d_from_str(initial_args.dataset):
    config_file = './model/PEMS03_{}.conf'.format(initial_args.model)
else:
    config_file = './model/{}_{}.conf'.format(initial_args.dataset, initial_args.model)
print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)
# print(config.sections())


second_parser = argparse.ArgumentParser(description="Parse additional arguments based on dataset", parents=[initial_parser],
                                        add_help=False)
#data
second_parser.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
second_parser.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
second_parser.add_argument('--lag', default=config['data']['lag'], type=int)
second_parser.add_argument('--horizon', default=config['data']['horizon'], type=int)

if parse_d_from_str(initial_args.dataset) == "w12":
    second_parser.add_argument('--num_nodes', default=12, type=int)
elif parse_d_from_str(initial_args.dataset):
    second_parser.add_argument('--num_nodes', default=parse_d_from_str(initial_args.dataset), type=int)
else:
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

# Add nb_blocks parameter for ASTGCN model
if 'nb_blocks' in config['model']:
    second_parser.add_argument('--nb_blocks', default=config['model']['nb_blocks'], type=int)
else:
    second_parser.add_argument('--nb_blocks', default=2, type=int)  # Default value if not in config

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
if config['train']['real_value']:
    print('use real value (no scaler)')
else:
    print('use scaler')

#test
second_parser.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
second_parser.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
second_parser.add_argument('--log_dir', default='./log/', type=str)
second_parser.add_argument('--log_step', default=config['log']['log_step'], type=int)
second_parser.add_argument('--plot', default=config['log']['plot'], type=eval)

if initial_args.dataset=='syn_gpvar':
    second_parser.add_argument('--adj_m', default = config['var_para']['adj_m'], type=str)
    second_parser.add_argument('--cor_m', default = config['var_para']['cor_m'], type=str)
    second_parser.add_argument('--cor_hop', default = config['var_para']['cor_hop'], type=int)
    second_parser.add_argument('--cor_t', default = config['var_para']['cor_t'], type=int)
    second_parser.add_argument('--Theta', default = config['var_para']['Theta'], type=str)
    second_parser.add_argument('--noise_mu', default = config['var_para']['noise_mu'], type=str)
    second_parser.add_argument('--noise_sigma', default = config['var_para']['noise_sigma'], type=str)
elif initial_args.dataset=='syn_tailup':
    from model.BasicTrainer import convert_str_2_tensor
    D_arr = config['var_para']['D']
    num_nodes = config['data']['num_nodes']
    cor_m_shape = (num_nodes, num_nodes)
    D = convert_str_2_tensor(D_arr, cor_m_shape, initial_args.device)
    second_parser.add_argument('--D', default= D, type=str)
    second_parser.add_argument('--alpha_true', default=config['var_para']['alpha_true'], type=str)
    second_parser.add_argument('--sigma2_true', default=config['var_para']['sigma2_true'], type=str)
    second_parser.add_argument('--phi_true', default=config['var_para']['phi_true'], type=str)
    second_parser.add_argument('--Sigma_spatial', default=config['var_para']['Sigma_spatial'], type=str)
    second_parser.add_argument('--num_nodes', default=20, type=int)
else:
    if parse_d_from_str(initial_args.dataset) == "w12":
        D_matrix_path = './data/PEMS03/PEMS03_w12_D.txt'
    elif parse_d_from_str(initial_args.dataset):
        D_matrix_path = './data/PEMS03/PEMS03_top_{}_D.txt'.format(parse_d_from_str(initial_args.dataset))
    elif initial_args.dataset == 'PEMSBAY':
        D_matrix_path = './data/PEMSBAY/pems_bay_sub_D.txt'
    else:
        D_matrix_path = './data/{}/{}_D.txt'.format(initial_args.dataset, initial_args.dataset)
    D = np.loadtxt(D_matrix_path)
    print('loaded D')
    second_parser.add_argument('--D', default=D, type=str)

second_parser.add_argument('--lmbd', default = 0, type=float, help= "The weight of graph covariance matrix.")
second_parser.add_argument('--Cov_type', default = 'ellip', type=str)
second_parser.add_argument('--w', default=2, type=int)


#save model
second_parser.add_argument('--save_path', default='./saved_model/', type=str)
if initial_args.dataset=='syn_gpvar':
    second_parser.add_argument('--save_filename', default='syn_gpvar_{}_{}.pth'.format(initial_args.syn_seed, 'saved_model'), type=str)
elif initial_args.dataset=='syn_tailup':
    second_parser.add_argument('--save_filename', default='syn_tailup_{}_{}.pth'.format(initial_args.syn_seed, initial_args.model ), type=str)
else:
    second_parser.add_argument('--save_filename', default='{}_{}.pth'.format(initial_args.dataset, initial_args.model), type=str)


#correction
second_parser.add_argument('--map_dim', default=3, type=int)
second_parser.add_argument('--correct_epochs', default=100, type=int)
second_parser.add_argument('--correct_ratio', default=0.5, type=float)
second_parser.add_argument('--correctionmode', default='mlp', type=str)
second_parser.add_argument('--alpha', default=0.05,type=int)
second_parser.add_argument('--size_loss_weight', default=1, type=float)
second_parser.add_argument('--weight_type', default='fixed', type=str)


print(f"Final parsed dataset: {initial_args.dataset}")
args = second_parser.parse_args(remaining_argv)

vars(args).update(vars(initial_args))

#args = args.parse_args()
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

print("horizon",args.horizon,"num_nodes", args.num_nodes)


#load dataset
train_loader, cal_loader, test_loader, scaler,  std, mean = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=True)

args.scaler = scaler
if args.normalizer == 'None':
    args.std, args.mean = 1, 0
else:
    args.std, args.mean = torch.Tensor(std).float(), torch.Tensor(mean).float()
    print('std, mean: ', std, mean)
# ####
# if not args.real_value:
#     scaler = None
# ####
from lib.metrics import MAE_torch

#init model

def norm_weight(distances):
    # Create a mask for finite distances.
    distances = distances.float()
    finite_mask = torch.isfinite(distances).bool()

    # Compute standard deviation only over finite distances.
    if torch.any(finite_mask):
        std = torch.std(distances[finite_mask])
    else:
        std = torch.tensor(1.0, device=distances.device)

    # Prevent division by zero.
    if std.item() == 0:
        std = torch.tensor(1e-6, device=distances.device)

    # Allocate an output tensor.
    normalized_weights = torch.zeros_like(distances, dtype=torch.float)

    # Apply the Gaussian kernel for finite distances.
    normalized_weights[finite_mask] = torch.exp(- distances[finite_mask] ** 2 / (std ** 2))

    return normalized_weights

if args.model == 'AGCRN':
    model = AGCRN(args)
else:
    if parse_d_from_str(initial_args.dataset):
        d = parse_d_from_str(initial_args.dataset)
        if d == "w12":
            file_path = "data/PEMS03/G_w12.gpickle"
        else:
            file_path = f"data/PEMS03/G_sub_{d}.gpickle"
        with open(file_path, 'rb') as f:
            G = pickle.load(f)
        # G.remove_edges_from(list(nx.selfloop_edges(G)))

        data = from_networkx(G, group_edge_attrs=['weight'])
        distances = data.edge_attr.squeeze(1)
        distances = norm_weight(distances)
        # print(distances)
        data.edge_weight = torch.Tensor(distances)
        args.G = data
    else:
        raise ValueError('Not support dataset except PEMSO3 12 nodes')
    if args.model == 'DCRNN':
        model = RecurrentGCN(args)
    else:
        model = TemporalGNN(args)


model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)

#print_model_parameters(model, only_num=False)

# def masked_mae_loss(y_pred, y_true):
#
#     mask = (y_true != 0).float()
#     mask /= mask.mean()
#     loss = torch.abs(y_pred - y_true)
#     loss = loss * mask
#     # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
#     loss[loss != loss] = 0
#     return loss.mean()

def masked_mae_loss(args, mask_value):
    # scaler = args.scaler
    def loss(preds, labels):
        # if scaler:
        #     preds = scaler.inverse_transform(preds)
        #     labels = scaler.inverse_transform(labels)
        preds = torch.nan_to_num(preds, nan=0.0)
        labels = torch.nan_to_num(labels, nan=0.0)#.squeeze(1)
        if args.model == 'A3TGCN':
            labels = labels.squeeze(1)
        elif args.model == 'ASTGCN':
            # ASTGCN may have different output shape handling
            if labels.dim() > preds.dim():
                labels = labels.squeeze(1)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(args, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss(args).to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss(args).to(args.device)
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
current_time = datetime.now().strftime('%Y%m%d%H%M')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'logs', args.dataset, 'log')
args.log_dir = log_dir
print("log_dir: ", log_dir)

#start training
trainer = Trainer(model, loss, optimizer, train_loader, cal_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)
print("Current mode: ", args.mode)

# Save the model
model_path = os.path.join(args.save_path, args.save_filename)

if args.mode == 'train':
    trainer.train()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    print(model_path)
    torch.save(model.state_dict(), model_path)
    print('*************** Base Model saved successfully at {}'.format(model_path))


elif args.mode == 'calcorrection':

    model.load_state_dict(torch.load( model_path ,map_location=args.device))
    if args.correctionmode=="mlp":
       closs=train_cal_correction(model,args,cal_loader,scaler)
       loss_cpu = [t.cpu().detach().numpy() for t in closs]

    # if args.correctionmode == 'add_dropout':
    #     dropout_model = InferenceWithDropout(model, dropout_rate = 0.3)

    # elif args.correctionmode=='gnn':
    #    closs=train_cal_correction_gnn(model,args,cal_loader,train_loader,scaler)
    #closs=loss.cpu().numpy()

    #print(closs.type())

    # plt.plot(loss_cpu)
    # plt.savefig('./correction_loss_gnn_plot.png')

# elif args.mode == 'test_map':
#
#     print("Start test  with gamma: {}, use mapping: {}".format(args.gamma,args.ACI_MLP_test))
#     correctmodel=RKHSMapping(args.num_nodes,args.map_dim)
#     correctmodel_path = '{}map_dim{}_{}'.format(args.save_path,args.map_dim,args.save_filename)
#     # model_path = '{}{}'.format(args.save_path,args.save_filename)
#     correctmodel.load_state_dict(
#         torch.load( correctmodel_path,
#                    map_location=args.device))
#
#     #mapmodel = RKHSMapping(args.num_nodes,args.map_dim).to(args.device)
#     #mapmodel.load_state_dict(torch.load('{}map_{}'.format(args.save_path,args.save_filename),map_location=args.device))
#     #model.load_state_dict(torch.load(model_path,map_location=args.device))
#     print("Start testing: Load training model from {} and correction model{}".format( model_path,correctmodel_path) )
#
#     trainer.test_map(model, correctmodel,trainer.args, test_loader, scaler, trainer.logger)

elif args.mode == 'test_gt':


    model_gt = GT_Predictor(args)
    print(type(model), type(model_gt))
    trainer_gt = Trainer(model_gt, loss, optimizer, train_loader, cal_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)
    print("start test")
    trainer_gt.gt_test(model_gt, None, trainer_gt.args, test_loader, scaler, trainer_gt.logger)

elif args.mode == 'test':

    from torch.utils.data import ConcatDataset, DataLoader    # Combine validation and test datasets
    combined_dataset = ConcatDataset([cal_loader.dataset, test_loader.dataset])
    combined_data_loader = DataLoader(combined_dataset, batch_size=32, shuffle=False)

    # No trained model
    if args.dataset =='syn_tailup' or args.dataset =='syn_tailup_gen':
        model = None
    else:
        # Special handling for ADDGCN model which has dimension mismatch
        if args.model == 'ADDGCN':
            # Load with strict=False to ignore mismatched parameters
            state_dict = torch.load(model_path, map_location=args.device)
            # Fix for ADDGCN model which has output layer dimension mismatch
            model_dict = model.state_dict()
            # Filter out layers with mismatched sizes (particularly the final output layer)
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            # Load the filtered state dict
            model.load_state_dict(state_dict, strict=False)
            print("ADDGCN model loaded with partial state_dict (ignoring mismatched layers)")
        else:
            model.load_state_dict(torch.load(model_path, map_location=args.device))

    print(f"{args.dataset},{args.syn_seed},{args.tinit}, {args.lmbd},{args.gamma},{args.Cov_type}")

    # picp, eff = trainer.gt_test(model, None, trainer.args, test_loader, scaler, trainer.logger)
    picp_mean, eff_mean, eff_var = trainer.gt_test(model, None, trainer.args, combined_data_loader, scaler, trainer.logger)

    print(f"{args.dataset},{args.syn_seed},{args.tinit}, {args.lmbd}, {args.gamma},{args.Cov_type},{picp_mean}, {eff_mean},  {eff_var}")

'''
elif args.mode == 'testgnn':

    correctmodel=Network(args)
    correctmodel_path = '{}correction_gnn_{}_{}'.format(args.save_path,args.tinit,args.save_filename)
    # model_path = '{}{}'.format(args.save_path,args.save_filename)

    correctmodel.load_state_dict(torch.load( correctmodel_path ,map_location=args.device))
    model.load_state_dict(torch.load( model_path ,map_location=args.device))
    print("Start testing: Load saved model from {} and {}".format( correctmodel_path, model_path) )
    trainer.test(model, correctmodel,trainer.args, test_loader, scaler, trainer.logger)


'''

