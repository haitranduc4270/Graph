import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter, Linear
from torch_geometric.utils import add_self_loops
import numpy as np

# Learnable composition operator for combining node and relation embeddings
class LearnableComp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(3))  # for mult, add, corr
        self.dim = dim

    def forward(self, h, r):
        # Element-wise multiplication
        mult = h * r
        # Element-wise addition
        add = h + r
        # Circular correlation via FFT
        corr = torch.fft.ifft(torch.fft.fft(h) * torch.conj(torch.fft.fft(r))).real
        # Softmax over learnable weights
        weights = F.softmax(self.weights, dim=0)
        # Weighted sum of composition operators
        return weights[0] * mult + weights[1] * add + weights[2] * corr

# Single CompGCN layer
class CompGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, dropout=0.3):
        super().__init__(aggr='add')
        # Relation embeddings (for both directions)
        self.rel_emb = nn.Embedding(num_relations * 2, in_channels)
        # Linear transformation for relations and nodes
        self.rel_linear = Linear(in_channels, out_channels)
        self.node_linear = Linear(in_channels, out_channels)
        # Learnable composition operator
        self.compose = LearnableComp(in_channels)
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization for parameters
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.rel_linear.weight)
        nn.init.xavier_uniform_(self.node_linear.weight)

    def forward(self, x, edge_index, edge_type):
        # Add self-loops to the graph
        edge_index, edge_type = add_self_loops(edge_index, edge_type, fill_value=0)

        # Get relation embeddings for edges
        rel = self.rel_emb(edge_type)
        # Message passing
        out = self.propagate(edge_index, x=x, rel=rel)
        # Linear transformation for node features
        out = self.node_linear(out)

        # Residual connection, normalization, activation, and dropout
        out = self.norm(out + x)
        out = F.relu(out)
        out = self.dropout(out)

        # Update relation embeddings
        updated_rel = self.rel_linear(self.rel_emb.weight)
        return out, updated_rel

    def message(self, x_j, rel):
        # Compose neighbor node and relation embeddings
        return self.compose(x_j, rel)

# Two-layer CompGCN model
class CompGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, embedding_dim, scoring='distmult', dropout=0.3):
        super().__init__()
        # Node and relation embeddings
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.rel_embedding = nn.Embedding(num_relations * 2, embedding_dim)

        # Two CompGCN layers
        self.conv1 = CompGCNLayer(embedding_dim, embedding_dim, num_relations, dropout=dropout)
        self.conv2 = CompGCNLayer(embedding_dim, embedding_dim, num_relations, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.scoring = scoring
        self.embedding_dim = embedding_dim

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization for embeddings
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)

    def forward(self, x, edge_index, edge_type, drop_edge_rate=0.1):
        # Get node embeddings
        x = self.embedding(x)

        # Edge dropout for regularization
        if self.training and drop_edge_rate > 0:
            mask = torch.rand(edge_index.size(1)) > drop_edge_rate
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]

        # Pass through two CompGCN layers
        x, _ = self.conv1(x, edge_index, edge_type)
        x, rels = self.conv2(x, edge_index, edge_type)
        return x, rels

    # Scoring function for link prediction
    def get_score(self, h, r, t):
        if self.scoring == "distmult":
            # DistMult scoring
            return torch.sum(h * r * t, dim=-1)
        elif self.scoring == "transe":
            # TransE scoring
            return -torch.norm(h + r - t, dim=-1)
        else:
            raise ValueError(f"Unsupported scoring: {self.scoring}")
