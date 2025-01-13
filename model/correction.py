import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
from lib.TrainInits import print_model_parameters
from model.AGCRN import AGCRN as Network
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class RKHSMapping(nn.Module):
    def __init__(self, num_nodes, low_dim, hidden_dim=64):
        super(RKHSMapping, self).__init__()
        # Define a multi-layer perceptron (MLP) for nonlinear mapping
        self.mlp = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),  # First layer from input dimension to hidden dimension
            nn.ReLU(),                         # Non-linear activation
            nn.Linear(hidden_dim, hidden_dim),  # Second layer (optional)
            nn.ReLU(),
            nn.Linear(hidden_dim, low_dim)      # Final layer to map to low_dim space
        )
        self.decode=nn.Sequential(
            nn.Linear(low_dim, hidden_dim),  # First layer from input dimension to hidden dimension
            nn.ReLU(),                         # Non-linear activation
            nn.Linear(hidden_dim, num_nodes))
        

    def forward(self, x):
        # Reshape if necessary: (B, 1, N, 1) -> (B, N)
        x = x.squeeze(-1).squeeze(1)  # Shape (B, N)
        
        # Pass through MLP to get low-dimensional embedding
        low_dim_embedding = self.mlp(x)  # Shape: (B, low_dim)
        x_recon=self.decode(low_dim_embedding)
        return low_dim_embedding, x_recon
def dot_product_distance_loss(pred, target):
    """
    pred: 模型的预测低维向量，形状为 (B, low_dim)
    target: 真实的低维向量，形状为 (B, low_dim)
    返回点积距离作为损失。
    """
    dot_product = torch.sum(pred * target, dim=-1)  # 计算每个样本的点积距离

    return dot_product


def plot_in_out_compare(pred , target, batch_idx, type):
    pred, target = pred.detach().cpu().numpy(), target.detach().cpu().numpy()
    X = np.vstack([pred, target])
    labels = np.array(['Reconstruct'] * len(pred) + ['GT'] * len(target))

    # PCA
    print('pca')
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # t-SNE
    print('t-sne')
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    # Plotting
    plt.figure(figsize=(12, 6))

    # PCA plot
    plt.subplot(1, 2, 1)
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label)
    plt.title("PCA")
    plt.legend()

    # t-SNE plot
    plt.subplot(1, 2, 2)
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label)
    plt.title("t-SNE")
    plt.legend()

    plt.show()
    plt.savefig("plot_mapping/plot_mapping_{}_{}.png".format(type,batch_idx))

def train_cal_correction(model,args,data_loader,scaler,tinit=300,graph_reg='False'):
   
    mapmodel = RKHSMapping(args.num_nodes,args.map_dim).to(args.device)

    optimizer = torch.optim.Adam(mapmodel.parameters(), weight_decay=5e-4, lr=0.0001)  
    pred_loss_hist, size_loss_hist, cons_loss_hist, val_size_loss_hist = [], [], [], []
    best_size_loss = 10000
    best_val_acc = 0
    y_true=[]
    y_pred=[]
    mapmodel.train()
    alpha=args.alpha
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                #print("data size is {}".format(data.size()))
                label = target[..., :args.output_dim]
                output,latent = model(data, teacher_forcing_ratio=0)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))

        if args.real_value:
                   y_pred = torch.cat(y_pred, dim=0)
        else:
                   y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        print('Starting mapping correction...')


    for epoch in range(1, args.correct_epochs + 1):  
          
            for batch_idx, (data, target) in enumerate(data_loader):

                optimizer.zero_grad()
                data = data[..., :args.input_dim]
                #print("data size is {}".format(data.size()))
                label = target[..., :args.output_dim]
                with torch.no_grad():
                     output,_ =model(data, target, teacher_forcing_ratio=0)#output_shape[B,1,num_node,1] latent_shape[B,1,num_node,hidden]
                     
                if args.real_value:
                   label = scaler.inverse_transform(label)
                map_pre,output_recon=mapmodel(output)
                map_true,true_recon=mapmodel(label)
                output, label = output.squeeze(-1).squeeze(1), label.squeeze(-1).squeeze(1)

                if batch_idx%5==0:
                    print(batch_idx)
                    plot_in_out_compare(output_recon, output, batch_idx, 'Pred')
                    plot_in_out_compare(true_recon, label, batch_idx, 'True')

                recon_loss=( dot_product_distance_loss(output_recon-output ,
                                                     output_recon-output)
                            +dot_product_distance_loss(true_recon-label,
                                                       true_recon-label) )
                weight=1
                
                loss=torch.mean(recon_loss)#torch.mean(dot_product_distance_loss(map_pre-map_true,map_pre-map_true)+weight*)
                pred_loss_hist.append(loss)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch}/{args.correct_epochs}], Loss: {loss.item():.4f}")
    torch.save(mapmodel.state_dict(), '{}map_dim{}_{}'.format(args.save_path,args.map_dim,args.save_filename))
    print("correction model saved!")

    return pred_loss_hist

    


def train_cal_correction_gnn(model,args,data_loader,scaler,conf_correct_model='mlp',tinit=300):




    pass