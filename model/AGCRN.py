import torch
import torch.nn as nn
from model.AGCRNCell import AGCRNCell
from model.BasicTrainer import convert_str_2_tensor
from scipy.stats import multivariate_normal

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.dropout_rate = 0.3
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets = None, teacher_forcing_ratio=0):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        # output: B, 1, N, 1(D)
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = self.dropout_layer(output)
        latent = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((latent))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output,latent


class AGCRN_LinkPrediction(nn.Module):
    def __init__(self, args):
        super(AGCRN_LinkPrediction, self).__init__()
        # Initialize AGCRN with given args
        self.agcrn = AGCRN(args)

    def forward(self, source, targets, node_pairs, teacher_forcing_ratio=0.5):
        """
        source: Input sequence (B, T_1, N, D)
        targets: Target sequence (B, T_2, N, D)
        node_pairs: List of node pairs (i, j) for link prediction
        teacher_forcing_ratio: Teacher forcing ratio for training
        """
        # Forward pass through AGCRN to get node embeddings
        init_state = self.agcrn.encoder.init_hidden(source.shape[0])
        output, hidden_states = self.agcrn.encoder(source, init_state, self.agcrn.node_embeddings)

        # Extract the final hidden state for each node
        final_latent_vectors = hidden_states[-1]  # Shape: (B, N, hidden_dim)

        # Link prediction
        link_logits = []
        for (i, j) in node_pairs:
            # Using dot product for similarity
            sim = torch.sum(final_latent_vectors[:, i, :] * final_latent_vectors[:, j, :], dim=-1)
            link_logits.append(sim)

        link_logits = torch.stack(link_logits, dim=1)  # Shape: (B, num_node_pairs)
        
        return link_logits


####### TODO

"""
    :param N: Number of nodes
    :param L: Number of correlated hops
    :param Q: Number of correlated observations in the past
    :param T: Number of time steps
    :param S: Graph shift operator
    :param Theta: Parameters of correlation scale: (L+1)*Q shape
"""
class GT_Predictor():
    def __init__(self, args):

        self.N = args.num_nodes
        self.S = convert_str_2_tensor( args.cor_m, (self.N, self.N), args.device)
        self.Q = args.cor_t
        self.L = args.cor_hop
        self.Theta = convert_str_2_tensor(args.Theta, (self.L+1, self.Q), args.device)  # cor_hop * cor_t
        self.noise_mu = convert_str_2_tensor(args.noise_mu,  self.N ,args.device).cpu().numpy()
        self.noise_sigma = convert_str_2_tensor( args.noise_sigma, (self.N, self.N), args.device).cpu().numpy()
        self.dist = multivariate_normal(mean=self.noise_mu, cov=self.noise_sigma)
        self.device=args.device

    def predict(self, x, target, teacher_forcing_ratio = 0):
        """
            input: Input sequence (batch size, time, number of node, 1)
            output:  (batch size, 1, number of node, 1)
        """
        B = x.shape[0]
        Res = torch.zeros( B, 1, self.N).to(self.device)
        x = x.squeeze(-1)
        #print("x: ", x.shape, x[0])

        for b in range(B):
            sum_term = torch.zeros(self.N).to(self.device)
            for l in range(self.L + 1):
                for q in range( self.Q):

                    shifted_x = (torch.matrix_power(self.S, l).float() @ x[b, - q - 1, : ]).to(self.device)
                    sum_term += self.Theta[l, q] * shifted_x
            noise = torch.tensor(self.dist.rvs(size=1), dtype=torch.float32, device=self.device)  # Generate noise
            Res[b, 0, :] = torch.tanh(sum_term) + 0.1 * noise
            
        # print('pred', Res[ 0, 0,:], 'target',target[0], '==')

        return Res.unsqueeze(-1)


class InferenceWithDropout(nn.Module):
    def __init__(self, model, dropout_rate):
        super(InferenceWithDropout, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        x,_ = self.model(input)
        x = self.dropout(x)
        return x