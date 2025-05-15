import torch
torch.manual_seed(1)
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import DCRNN, BatchedDCRNN
from torch_geometric_temporal.nn.attention import ASTGCN
from torch_geometric.nn import ChebConv, GCNConv
import torchdiffeq


class STPredictor(torch.nn.Module):
    """
    Base class for Spatial-Temporal Graph Neural Network models.
    This class provides a unified interface for different ST-GNN models.
    """
    def __init__(self, args):
        super(STPredictor, self).__init__()
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.edge_index = args.G.edge_index
        self.edge_weight = args.G.edge_weight.float()
        self.batch_size = args.batch_size
        self.args = args
        self.model_type = args.model
        self.device = args.device

        # Initialize appropriate model based on model type
        self._build_model()
        
    def _build_model(self):
        """
        Build the specific model architecture.
        This method should be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement _build_model method")
    
    def forward(self, X):
        """
        Forward pass of the model.
        This method should be implemented by all subclasses.
        
        Args:
            X: Input tensor with shape [batch_size, seq_length, num_nodes, features]
            
        Returns:
            Prediction tensor and optional auxiliary outputs
        """
        raise NotImplementedError("Subclasses must implement forward method")
        
    def to(self, device):
        """
        Moves the model to the specified device and ensures graph data is also on that device
        """
        # Move the model to the device
        super().to(device)
        # Move edge_index and edge_weight to the same device
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)
        return self


class DCRNNModel(STPredictor):
    """
    Diffusion Convolutional Recurrent Neural Network model
    """
    def _build_model(self):
        # Use BatchedDCRNN for batched processing
        self.recurrent = BatchedDCRNN(
            in_channels=self.input_dim, 
            out_channels=self.output_dim, 
            K=3
        )
        
        # Output projection layer
        self.linear = torch.nn.Linear(self.hidden_dim, self.output_dim)
        
        # Optional dropout
        self.dropout_rate = 0.3
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def forward(self, X):
        # X shape: [batch_size, seq_length, num_nodes, features]
        # Ensure all inputs are on the same device
        device = X.device
        edge_index = self.edge_index.to(device)
        edge_weight = self.edge_weight.to(device)
        
        # Process through DCRNN
        output = self.recurrent(X, edge_index, edge_weight)
        
        # Take the last time step output
        h = output[:, -1:, :, :]
        
        return h, None


class ASTGCNModel(STPredictor):
    """
    Attention-based Spatial-Temporal Graph Convolutional Network model
    """
    def _build_model(self):
        # Configure ASTGCN specific parameters
        self.time_strides = 1
        self.nb_chev_filter = self.hidden_dim
        self.nb_time_filter = self.hidden_dim
        
        # ASTGCN implementation with correct parameter names
        self.tgnn = ASTGCN(
            nb_block=self.args.nb_blocks,
            in_channels=self.input_dim,
            K=3,  # Chebyshev filter size
            nb_chev_filter=self.nb_chev_filter,
            nb_time_filter=self.nb_time_filter,
            time_strides=self.time_strides,
            num_for_predict=self.horizon,
            len_input=self.args.lag,  # Sequence length
            num_of_vertices=self.num_nodes
        )

    def forward(self, X):
        # ASTGCN expects input shape: [batch_size, num_nodes, features, seq_length]
        input_transformed = X.permute(0, 2, 3, 1)
        
        # Ensure all inputs are on the same device
        device = X.device
        edge_index = self.edge_index.to(device)
        
        # ASTGCN returns predictions directly
        h = self.tgnn(input_transformed, edge_index)
        
        # Reshape output to match expected format: [batch_size, 1, num_nodes, output_dim]
        # The shape from ASTGCN is likely [batch_size, num_nodes, horizon]
        batch_size = h.shape[0]
        
        # Add a time dimension and reorder to [batch_size, horizon, num_nodes, output_dim]
        h = h.unsqueeze(-1)  # Add output_dim
        h = h.permute(0, 2, 1, 3)  # [batch_size, horizon, num_nodes, output_dim]
        
        # Only return the first prediction step to match other models 
        # that only predict one step ahead
        h = h[:, :1, :, :]
        
        return h, None


class STGCNModel(STPredictor):
    """
    Spatio-Temporal Graph Convolutional Network model
    """
    def _build_model(self):
        # Configure ST-GCN specific parameters
        self.dropout_rate = 0.3
        
        # Define the graph convolutional layers
        self.gc1 = GCNConv(self.input_dim, self.hidden_dim)
        self.gc2 = GCNConv(self.hidden_dim, self.hidden_dim)
        
        # Define temporal convolutional layers - corrected dimensions
        # For Conv1d: (in_channels, out_channels, kernel_size)
        self.tc1 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.tc2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, X):
        # X shape: [batch_size, seq_length, num_nodes, features]
        batch_size, seq_length, num_nodes, features = X.shape
        
        # Ensure all inputs are on the same device
        device = X.device
        edge_index = self.edge_index.to(device)
        edge_weight = self.edge_weight.to(device)
        
        # Reshape for graph convolution
        # Process each time step through graph convolution
        spatial_features = []
        for t in range(seq_length):
            x_t = X[:, t, :, :].reshape(-1, features)  # [batch_size*num_nodes, features]
            
            # First graph convolution
            h = self.gc1(x_t, edge_index, edge_weight)
            h = F.relu(h)
            h = self.dropout(h)
            
            # Second graph convolution
            h = self.gc2(h, edge_index, edge_weight)
            h = F.relu(h)
            h = self.dropout(h)
            
            # Reshape back to [batch_size, num_nodes, hidden_dim]
            h = h.reshape(batch_size, num_nodes, self.hidden_dim)
            spatial_features.append(h)
        
        # Stack spatial features along time dimension [batch_size, seq_length, num_nodes, hidden_dim]
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # Process each node separately through the temporal conv layers
        temporal_outputs = []
        for n in range(num_nodes):
            # Extract features for node n: [batch_size, seq_length, hidden_dim]
            node_features = spatial_features[:, :, n, :]
            
            # Reshape for Conv1d which expects [batch, channels, seq_len]
            # Here channels = hidden_dim, seq_len = seq_length
            node_features = node_features.transpose(1, 2)  # [batch_size, hidden_dim, seq_length]
            
            # Apply temporal convolutions
            t_out = self.tc1(node_features)
            t_out = F.relu(t_out)
            t_out = self.dropout(t_out)
            
            t_out = self.tc2(t_out)
            t_out = F.relu(t_out)
            t_out = self.dropout(t_out)
            
            # The output has shape [batch_size, hidden_dim, seq_length]
            # Take the last time step
            t_out = t_out[:, :, -1]  # [batch_size, hidden_dim]
            
            # Output projection for this node
            node_out = self.output_layer(t_out)  # [batch_size, output_dim]
            temporal_outputs.append(node_out)
        
        # Stack outputs for all nodes [batch_size, num_nodes, output_dim]
        out = torch.stack(temporal_outputs, dim=1)
        
        # Add time dimension to match expected output format [batch_size, 1, num_nodes, output_dim]
        out = out.unsqueeze(1)
        
        return out, None


class STGODEModel(STPredictor):
    """
    Spatial-Temporal Graph Ordinary Differential Equation model
    
    References:
    - Paper: Spatiotemporal Neural ODE Networks for Traffic Flow Forecasting
    - Link: https://ojs.aaai.org/index.php/AAAI/article/view/20291
    """
    def _build_model(self):
        # Configure STGODE specific parameters
        self.input_dim = self.args.input_dim
        self.output_dim = self.args.output_dim
        self.hidden_dim = self.args.rnn_units
        self.num_nodes = self.args.num_nodes
        self.dropout_rate = 0.3
        
        # Spatial encoding and decoding modules
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # ODE-specific layers - create ODEFunc for evolving states
        self.ode_func = ODEFunc(
            hidden_dim=self.hidden_dim,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            num_nodes=self.num_nodes
        )
        
        # Define the diffusion GCN layer - encode spatial dependencies on input
        self.gcn = GCNConv(self.hidden_dim, self.hidden_dim)

    def forward(self, X):
        # X shape: [batch_size, seq_length, num_nodes, features]
        batch_size, seq_length, num_nodes, features = X.shape
        device = X.device
        
        # Move graph data to the right device
        edge_index = self.edge_index.to(device)
        edge_weight = self.edge_weight.to(device)
        
        # Extract the last time step for initial condition
        x_t = X[:, -1, :, :]  # [batch_size, num_nodes, features]
        
        # Flatten nodes and batch for transformation
        x_flat = x_t.reshape(-1, features)  # [batch_size*num_nodes, features]
        
        # Initial encoding to hidden dimension
        h0 = self.encoder(x_flat)  # [batch_size*num_nodes, hidden_dim]
        
        # Apply graph convolution to capture spatial dependencies
        h0 = self.gcn(h0, edge_index, edge_weight)  # [batch_size*num_nodes, hidden_dim]
        h0 = F.relu(h0)
        
        # Reshape back to [batch_size, num_nodes, self.hidden_dim]
        h0 = h0.reshape(batch_size, num_nodes, self.hidden_dim)
        
        # Update ODE function with current edge data
        self.ode_func.edge_index = edge_index
        self.ode_func.edge_weight = edge_weight
        
        # Set time horizon based on prediction horizon
        t_span = torch.tensor([0., float(self.horizon)]).to(device)
        
        try:
            # First attempt with adaptive solver
            ode_solution = torchdiffeq.odeint(
                self.ode_func,
                h0,
                t_span,
                method='dopri5',
                rtol=1e-3,
                atol=1e-3,
                options={'max_num_steps': 2000}
            )
        except Exception as e:
            print(f"Adaptive solver failed: {e}, falling back to fixed-step method")
            # Fall back to fixed solver 
            ode_solution = torchdiffeq.odeint(
                self.ode_func,
                h0,
                t_span,
                method='rk4',
                options={'step_size': 0.05}
            )
            
        # Extract final state
        h_t = ode_solution[-1]  # [batch_size, num_nodes, hidden_dim]
        
        # Reshape for decoding
        h_flat = h_t.reshape(-1, self.hidden_dim)  # [batch_size*num_nodes, hidden_dim]
        
        # Decode to output dimension
        out_flat = self.decoder(h_flat)  # [batch_size*num_nodes, output_dim]
        
        # Reshape to expected format [batch_size, num_nodes, output_dim]
        out = out_flat.reshape(batch_size, num_nodes, self.output_dim)
        
        # Add time dimension to match expected output format [batch_size, 1, num_nodes, output_dim]
        out = out.unsqueeze(1)
        
        return out, None


class ODEFunc(nn.Module):
    """
    Neural ODE function for STGODE
    
    This class defines the ODE dynamics for evolving node states over time.
    The dynamics incorporate both graph structure and nonlinear transformations.
    """
    def __init__(self, hidden_dim, edge_index, edge_weight, num_nodes):
        super(ODEFunc, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes
        
        # Neural networks for node-wise dynamics
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolution for spatial interactions
        self.gc1 = GCNConv(hidden_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)
        
        # Gating mechanism
        self.gating = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, t, x):
        """
        Defines the dynamics dx/dt = f(t, x)
        
        Args:
            t: Current time point (scalar)
            x: Current state [batch_size, num_nodes, hidden_dim]
            
        Returns:
            dx/dt: State derivative [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, hidden_dim = x.shape
        
        # Reshape for graph operations
        x_flat = x.reshape(-1, hidden_dim)  # [batch_size*num_nodes, hidden_dim]
        
        # Graph convolution path
        h_graph = self.gc1(x_flat, self.edge_index, self.edge_weight)
        h_graph = F.relu(h_graph)
        h_graph = self.gc2(h_graph, self.edge_index, self.edge_weight)
        
        # Node-wise MLP path  
        h_node = self.node_mlp(x_flat)
        
        # Combine graph and node representations with gating
        h_combined = torch.cat([h_graph, h_node], dim=1)
        gate = self.gating(h_combined)
        
        # Final derivative as gated combination
        dx = gate * h_graph + (1 - gate) * h_node
        
        # Reshape back to match input shape
        dx = dx.reshape(batch_size, num_nodes, hidden_dim)
        
        return dx


class MSTGCNModel(STPredictor):
    """
    Multi-Component Spatial-Temporal Graph Convolutional Network model
    
    MSTGCN processes spatial and temporal components with multiple parallel GCN blocks.
    It captures the spatial dependencies through graph convolution while modeling temporal
    patterns with 1D convolutions.
    """
    def _build_model(self):
        # Model hyperparameters
        self.input_dim = self.args.input_dim
        self.output_dim = self.args.output_dim
        self.hidden_dim = self.args.rnn_units
        self.num_nodes = self.args.num_nodes
        self.seq_len = self.args.lag
        self.horizon = self.args.horizon
        self.dropout_rate = 0.3
        self.num_blocks = self.args.nb_blocks if hasattr(self.args, 'nb_blocks') else 3
        
        # Spatial embedding for nodes
        self.node_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.hidden_dim), requires_grad=True
        )
        
        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )
        
        # Multi-component GCN blocks
        self.st_blocks = nn.ModuleList([
            STBlock(
                hidden_dim=self.hidden_dim,
                num_nodes=self.num_nodes,
                cheb_k=self.args.cheb_k
            ) for _ in range(self.num_blocks)
        ])
        
        # Output projection layers
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=(1, self.seq_len)),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.horizon * self.output_dim, kernel_size=(1, 1))
        )
        
    def build_adj_matrix(self):
        """
        Constructs the adjacency matrix from node embeddings
        """
        node_embeddings = self.node_embeddings  # [num_nodes, hidden_dim]
        
        # Compute self-adaptive adjacency matrix
        adj = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        
        # Create Chebyshev polynomial basis
        adj_list = [torch.eye(self.num_nodes).to(node_embeddings.device)]
        adj_list.append(adj)
        
        for k in range(2, self.args.cheb_k):
            # Recursive computation of Chebyshev polynomials
            adj_k = 2 * torch.matmul(adj, adj_list[-1]) - adj_list[-2]
            adj_list.append(adj_k)
            
        return torch.stack(adj_list, dim=0)  # [cheb_k, num_nodes, num_nodes]

    def forward(self, X):
        # X shape: [batch_size, seq_length, num_nodes, features]
        batch_size, seq_length, num_nodes, features = X.shape
        device = X.device
        
        # Ensure node embeddings are on correct device
        self.node_embeddings = self.node_embeddings.to(device)
        
        # Build adjacency matrix
        adj_mx = self.build_adj_matrix()  # [cheb_k, num_nodes, num_nodes]
        
        # Reshape for 2D convolution: [batch_size, features, num_nodes, seq_length]
        x = X.permute(0, 3, 2, 1)
        
        # Input projection
        x = self.input_proj(x)  # [batch_size, hidden_dim, num_nodes, seq_length]
        
        # Process through ST blocks
        for st_block in self.st_blocks:
            x = st_block(x, adj_mx)
        
        # Output projection
        out = self.output_proj(x)  # [batch_size, horizon*output_dim, num_nodes, 1]
        
        # Reshape to final output format
        out = out.reshape(batch_size, self.horizon, self.output_dim, num_nodes)
        out = out.permute(0, 1, 3, 2)  # [batch_size, horizon, num_nodes, output_dim]
        
        # Return only the first time step if required
        if self.horizon > 1 and self.args.model != 'MSTGCN':
            out = out[:, :1, :, :]
        
        return out, None


class STBlock(nn.Module):
    """
    Spatial-Temporal Block for MSTGCN
    """
    def __init__(self, hidden_dim, num_nodes, cheb_k):
        super(STBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.cheb_k = cheb_k
        
        # Spatial component: Chebychev GCN
        self.spatial_filter = ChebFilterBlock(hidden_dim, hidden_dim, cheb_k)
        
        # Temporal component: 1D convolution along the time axis
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Gate mechanism
        self.gate_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=(1, 1)),
            nn.Sigmoid()
        )
        
        # Final projection
        self.residual_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        
        # Use BatchNorm2d instead of LayerNorm with None dimension
        self.batch_norm = nn.BatchNorm2d(hidden_dim)
        
    def forward(self, x, adj_mx):
        """
        Args:
            x: Input features [batch_size, hidden_dim, num_nodes, seq_length]
            adj_mx: Adjacency matrix [cheb_k, num_nodes, num_nodes]
        """
        batch_size, hidden_dim, num_nodes, seq_length = x.shape
        
        # Spatial component
        spatial_out = self.spatial_filter(x, adj_mx)  # [batch_size, hidden_dim, num_nodes, seq_length]
        
        # Temporal component
        temporal_out = self.temporal_conv(x)  # [batch_size, hidden_dim, num_nodes, seq_length]
        
        # Gate mechanism
        combined = torch.cat([spatial_out, temporal_out], dim=1)  # [batch_size, 2*hidden_dim, num_nodes, seq_length]
        gate = self.gate_conv(combined)  # [batch_size, hidden_dim, num_nodes, seq_length]
        
        # Combine components with gating and add residual connection
        x_residual = self.residual_conv(x)
        out = gate * spatial_out + (1 - gate) * temporal_out + x_residual
        
        # Apply batch normalization instead of layer normalization
        out = self.batch_norm(out)
        
        return out


class ChebFilterBlock(nn.Module):
    """
    Chebyshev graph convolution block for MSTGCN
    """
    def __init__(self, in_channels, out_channels, K):
        super(ChebFilterBlock, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Weight matrix for each Chebyshev order
        self.weights = nn.Parameter(
            torch.Tensor(K, in_channels, out_channels),
            requires_grad=True
        )
        self.bias = nn.Parameter(
            torch.Tensor(out_channels),
            requires_grad=True
        )
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj_mx):
        """
        Args:
            x: Input features [batch_size, in_channels, num_nodes, seq_length]
            adj_mx: Adjacency matrix [K, num_nodes, num_nodes]
        """
        batch_size, in_channels, num_nodes, seq_length = x.shape
        
        # Reshape for spatial convolution
        x_reshape = x.permute(0, 3, 1, 2)  # [batch_size, seq_length, in_channels, num_nodes]
        x_reshape = x_reshape.reshape(-1, in_channels, num_nodes)  # [batch_size*seq_length, in_channels, num_nodes]
        
        # Apply Chebyshev graph convolution
        output = torch.zeros(batch_size * seq_length, self.out_channels, num_nodes, device=x.device)
        
        for k in range(self.K):
            # Filter by adjacency matrix
            adj_k = adj_mx[k]  # [num_nodes, num_nodes]
            x_k = torch.matmul(x_reshape, adj_k)  # [batch_size*seq_length, in_channels, num_nodes]
            
            # Apply weights for this order
            w_k = self.weights[k]  # [in_channels, out_channels]
            x_k = torch.einsum('bim,io->bom', x_k, w_k)  # [batch_size*seq_length, out_channels, num_nodes]
            
            # Accumulate result
            output += x_k
            
        # Add bias
        output += self.bias.view(1, -1, 1)
        
        # Reshape back to original format
        output = output.reshape(batch_size, seq_length, self.out_channels, num_nodes)
        output = output.permute(0, 2, 3, 1)  # [batch_size, out_channels, num_nodes, seq_length]
        
        return output


def create_model(args):
    """
    Factory function to create the appropriate model based on the specified type.
    
    Args:
        args: Configuration arguments including model type
        
    Returns:
        An instance of the specified model type
    """
    model_types = {
        'DCRNN': DCRNNModel,
        'ASTGCN': ASTGCNModel,
        'STGCN': STGCNModel,
        'STGODE': STGODEModel,
        'MSTGCN': MSTGCNModel,
    }
    
    # Get the model class based on the specified type
    ModelClass = model_types[args.model]
    return ModelClass(args)

# For backward compatibility with existing code
class RecurrentGCN(DCRNNModel):
    def __init__(self, args):
        super(RecurrentGCN, self).__init__(args)
        print('batch_size', args.batch_size)


class TemporalGNN(STPredictor):
    def __init__(self, args):
        super(TemporalGNN, self).__init__(args)
    
    def _build_model(self):
        # Determine the model type and build it
        if  self.model_type == 'ASTGCN':
            self._model = ASTGCNModel(self.args)
        elif self.model_type == 'STGCN':
            self._model = STGCNModel(self.args)
        elif self.model_type == 'STGODE':
            self._model = STGODEModel(self.args)
        elif self.model_type == 'MSTGCN':
            self._model = MSTGCNModel(self.args)
    
    def forward(self, X):
        return self._model.forward(X)