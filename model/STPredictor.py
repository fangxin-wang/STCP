
import torch
torch.manual_seed(1)
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import DCRNN,A3TGCN,A3TGCN2, BatchedDCRNN

# Ensure N is defined appropriately, for example:

class RecurrentGCN(torch.nn.Module):
    def __init__(self, args):
        super(RecurrentGCN, self).__init__()
        print('batch_size',args.batch_size)
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.recurrent = BatchedDCRNN(self.input_dim, self.output_dim, K = 3)
        self.linear = torch.nn.Linear(self.hidden_dim, args.output_dim)
        self.edge_index = args.G.edge_index
        self.edge_weight = args.G.edge_weight.float()
        self.batch_size = args.batch_size
        self.dropout_rate = 0.3
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        # self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)


    def forward(self, X):
        # source: B, T, N, D_i
        # input:  (batch_size, seq_length, num_nodes, num_features)
        output = self.recurrent(X, self.edge_index, self.edge_weight)
        h = output[:, -1:, :, :]
        # target: B, T, N, D_o
        # output: (batch_size, seq_length, num_nodes, out_channels)
        # output = self.dropout_layer(output)
        # print('latent',latent)

        return h, None


class TemporalGNN(torch.nn.Module):
    def __init__(self, args):
        super(TemporalGNN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.edge_index = args.G.edge_index
        self.edge_weight = args.G.edge_weight.float()
        self.batch_size = args.batch_size
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=self.input_dim,  out_channels=self.hidden_dim, periods=self.horizon ,batch_size=self.batch_size) # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(self.hidden_dim, self.horizon)

    def forward(self, X):
        # X: B, T, N, D_i
        # input:  (batch_size, seq_length, num_nodes, num_features)
        input = X.permute(0, 2, 3, 1 )
        h = self.tgnn(input, self.edge_index, self.edge_weight) # x [b, 207, 2, 12]  returns h [b, 207, 12]
        # print(input.shape, h.shape)
        h = F.relu(h)
        h = self.linear(h)
        # print(h.shape)
        return h, None