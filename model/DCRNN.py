import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DCRNNCell(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network Cell
    
    Args:
        node_num: Number of nodes in the graph
        dim_in: Input dimension
        dim_out: Output dimension
        K: Number of diffusion steps (order of Chebyshev polynomials)
        bias: Whether to include bias term
    """
    def __init__(self, node_num, dim_in, dim_out, K, bias=True):
        super(DCRNNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.K = K  # Number of diffusion steps
        
        # Parameters for the diffusion convolution
        self.weight_gate = nn.Parameter(torch.empty(2 * dim_out, self.K * (dim_in + dim_out)))
        self.weight_candidate = nn.Parameter(torch.empty(dim_out, self.K * (dim_in + dim_out)))
        
        if bias:
            self.bias_gate = nn.Parameter(torch.empty(2 * dim_out))
            self.bias_candidate = nn.Parameter(torch.empty(dim_out))
        else:
            self.register_parameter('bias_gate', None)
            self.register_parameter('bias_candidate', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_gate)
        nn.init.xavier_uniform_(self.weight_candidate)
        if self.bias_gate is not None:
            nn.init.zeros_(self.bias_gate)
            nn.init.zeros_(self.bias_candidate)
    
    def forward(self, inputs, hx, edge_index, edge_weight=None):
        """
        Args:
            inputs: Input features (batch_size, num_nodes, dim_in)
            hx: Hidden state (batch_size, num_nodes, hidden_dim)
            edge_index: Graph edge indices
            edge_weight: Edge weights
            
        Returns:
            Updated hidden state
        """
        # Perform diffusion convolution on concatenated input and hidden state
        x = torch.cat([inputs, hx], dim=-1)  # [B, N, dim_in + hidden_dim]
        
        # Compute diffusion convolution
        x_diffused = self._diffusion_conv(x, edge_index, edge_weight)  # [B, N, K * (dim_in + hidden_dim)]
        
        # Compute gates
        gates = torch.matmul(x_diffused, self.weight_gate.t())
        if self.bias_gate is not None:
            gates = gates + self.bias_gate
        
        # Split into reset and update gates
        z_r = torch.sigmoid(gates)
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        
        # Compute candidate state
        x_candidate = torch.cat([inputs, r * hx], dim=-1)
        x_candidate_diffused = self._diffusion_conv(x_candidate, edge_index, edge_weight)
        
        candidate = torch.matmul(x_candidate_diffused, self.weight_candidate.t())
        if self.bias_candidate is not None:
            candidate = candidate + self.bias_candidate
        
        # Apply non-linearity
        candidate = torch.tanh(candidate)
        
        # Update hidden state
        h_new = (1 - z) * hx + z * candidate
        
        return h_new
    
    def _diffusion_conv(self, x, edge_index, edge_weight=None):
        """
        Perform diffusion convolution with Chebyshev polynomials
        
        Args:
            x: Input tensor [B, N, F]
            edge_index: Edge indices
            edge_weight: Edge weights
            
        Returns:
            Diffusion convolution result [B, N, K*F]
        """
        batch_size, num_nodes, feat_dim = x.size()
        
        # Initialize with identity matrix for k=0 (no diffusion)
        results = [x]
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # Normalize adjacency matrix (edge_index and edge_weight represent sparse adjacency)
        row, col = edge_index
        edge_val = edge_weight
        
        # Compute powers of normalized adjacency matrix
        x_0 = x  # For k=0 (identity)
        x_1 = self._graph_conv(x_0, edge_index, edge_val)  # For k=1
        results.append(x_1)
        
        # Higher-order diffusions (if K > 2)
        for k in range(2, self.K):
            x_k = 2 * self._graph_conv(x_1, edge_index, edge_val) - x_0
            results.append(x_k)
            x_0, x_1 = x_1, x_k
        
        # Concatenate all diffusion results
        return torch.cat(results, dim=-1)
    
    def _graph_conv(self, x, edge_index, edge_weight):
        """
        Simple GCN-style graph convolution for sparse adjacency matrix
        
        Args:
            x: Input tensor [B, N, F]
            edge_index: Edge indices
            edge_weight: Edge weights
            
        Returns:
            Convolved tensor [B, N, F]
        """
        batch_size, num_nodes, feat_dim = x.size()
        
        # Reshape for efficient sparse matrix multiplication
        x_flat = x.reshape(-1, feat_dim)  # [B*N, F]
        
        # Convert to sparse format for computation
        row, col = edge_index
        
        # Apply convolution for each batch
        output = torch.zeros_like(x_flat)
        
        # For each node, aggregate features from neighbors
        for i in range(batch_size):
            node_offset = i * num_nodes
            # For each edge, aggregate weighted features
            for e in range(edge_index.size(1)):
                src, dst = col[e], row[e]
                output[node_offset + dst] += edge_weight[e] * x_flat[node_offset + src]
        
        # Reshape back to original shape
        output = output.reshape(batch_size, num_nodes, feat_dim)
        
        return output
    
    def init_hidden_state(self, batch_size):
        """
        Initialize hidden state with zeros
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial hidden state [batch_size, node_num, hidden_dim]
        """
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class DCRNN(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network
    
    Args:
        args: Model configuration arguments
    """
    def __init__(self, args):
        super(DCRNN, self).__init__()
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.K = args.cheb_k  # Diffusion steps K (Chebyshev order)
        
        # Dropout for regularization
        self.dropout_rate = 0.3
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        
        # Stack of DCRNN cells
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(DCRNNCell(self.num_nodes, self.input_dim, self.hidden_dim, self.K))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(DCRNNCell(self.num_nodes, self.hidden_dim, self.hidden_dim, self.K))
        
        # Output projection layer
        self.output_layer = nn.Conv2d(1, self.horizon * self.output_dim, 
                                      kernel_size=(1, self.hidden_dim), bias=True)
    
    def forward(self, source, edge_index, edge_weight=None, targets=None, teacher_forcing_ratio=0):
        """
        Forward pass through DCRNN
        
        Args:
            source: Input sequence [batch_size, seq_len, num_nodes, input_dim]
            edge_index: Graph edge indices
            edge_weight: Edge weights
            targets: Target sequence for teacher forcing
            teacher_forcing_ratio: Ratio of teacher forcing
            
        Returns:
            Predictions and hidden states
        """
        batch_size, seq_len, num_nodes, _ = source.size()
        
        # Initialize hidden states
        hidden_states = self.init_hidden(batch_size)
        
        # Encoder
        encoder_outputs = []
        for t in range(seq_len):
            current_inputs = source[:, t, :, :]
            for layer_num, dcrnn_cell in enumerate(self.dcrnn_cells):
                next_hidden = dcrnn_cell(current_inputs, hidden_states[layer_num], edge_index, edge_weight)
                hidden_states[layer_num] = next_hidden
                current_inputs = next_hidden
            
            encoder_outputs.append(current_inputs)
        
        # Stack encoder outputs
        encoder_outputs = torch.stack(encoder_outputs, dim=1)
        
        # Apply dropout
        encoder_outputs = self.dropout_layer(encoder_outputs)
        
        # Take last time step output as initial decoder input
        last_output = encoder_outputs[:, -1:, :, :]
        
        # Output projection
        output = self.output_layer(last_output.permute(0, 3, 2, 1))  # [B, horizon*output_dim, num_nodes, 1]
        output = output.squeeze(-1).reshape(batch_size, self.horizon, self.output_dim, num_nodes)
        output = output.permute(0, 1, 3, 2)  # [B, horizon, num_nodes, output_dim]
        
        return output, hidden_states
    
    def init_hidden(self, batch_size):
        """
        Initialize hidden states for all layers
        
        Args:
            batch_size: Batch size
            
        Returns:
            List of initial hidden states for each layer
        """
        hidden_states = []
        for cell in self.dcrnn_cells:
            hidden_states.append(cell.init_hidden_state(batch_size))
        return hidden_states
    
    def predict(self, source, edge_index, edge_weight=None, targets=None, teacher_forcing_ratio=0):
        """
        Generate predictions for compatibility with the rest of the codebase
        
        Args:
            source: Input sequence
            edge_index: Graph edge indices
            edge_weight: Edge weights
            targets: Target sequence (not used in this implementation)
            teacher_forcing_ratio: Teacher forcing ratio (not used in this implementation)
            
        Returns:
            Predictions
        """
        output, _ = self.forward(source, edge_index, edge_weight, targets, teacher_forcing_ratio)
        return output