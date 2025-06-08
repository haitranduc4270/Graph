import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, MessagePassing
from torch.nn import Linear, Parameter
from torch_geometric.utils import softmax

# Vanilla Relational Graph Convolutional Network (RGCN)
class RGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, embedding_dim, scoring='distmult', num_bases=30, dropout=0.3):
        super(RGCN, self).__init__()
        # Node and relation embeddings
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.rel_embedding = nn.Embedding(num_relations, embedding_dim)

        # Two RGCNConv layers
        self.conv1 = RGCNConv(embedding_dim, embedding_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(embedding_dim, embedding_dim, num_relations, num_bases=num_bases)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Scoring function type
        self.scoring = scoring

        # Xavier initialization
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)

    def forward(self, x, edge_index, edge_type):
        x = self.embedding(x)
        x = self.dropout(self.norm1(torch.relu(self.conv1(x, edge_index, edge_type))))
        x = self.dropout(self.norm2(torch.relu(self.conv2(x, edge_index, edge_type))))
        return x
    
    def get_score(self, h, r, t):
        # Link prediction scoring functions
        if self.scoring == "distmult":
            return torch.sum(h * r * t, dim=1)
        elif self.scoring == "transe":
            return -torch.norm(h + r - t, dim=1)
        else:
            raise ValueError("Unknown scoring function")

# RGCN with residual (skip) connections
class RGCN_Residual(RGCN):
    def __init__(self, num_entities, num_relations, embedding_dim, dropout=0.3, num_bases=30):
        super().__init__(num_entities, num_relations, embedding_dim, dropout=dropout, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        x = self.embedding(x)

        h1 = self.conv1(x, edge_index, edge_type)
        h1 = torch.relu(h1)
        h1 = self.norm1(h1)
        h1 = self.dropout(h1)
        h1 = h1 + x  # skip connection

        h2 = self.conv2(h1, edge_index, edge_type)
        h2 = torch.relu(h2)
        h2 = self.norm2(h2)
        h2 = self.dropout(h2)
        h2 = h2 + h1  # skip connection

        return h2


# Relational Graph Attention Convolution layer
class RGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, dropout=0.2):
        super(RGATConv, self).__init__(aggr='add')  # aggregation type

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.linear = Linear(in_channels, out_channels)
        self.rel_emb = nn.Embedding(num_relations, in_channels)
        self.rel_linear = Linear(in_channels, out_channels)

        # Attention parameters
        self.attn = Parameter(torch.Tensor(1, out_channels * 3))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.rel_linear.weight)

    def forward(self, x, edge_index, edge_type):
        x = self.linear(x)  # shape: [num_nodes, out_channels]
        rel = self.rel_linear(self.rel_emb(edge_type))  # shape: [num_edges, out_channels]
        return self.propagate(edge_index, x=x, rel=rel, edge_type=edge_type)

    def message(self, x_i, x_j, rel, index):
        # Concatenate for attention: [x_i, x_j, rel]
        z = torch.cat([x_i, x_j, rel], dim=-1)  # shape: [num_edges, 3*out_channels]
        alpha = self.leaky_relu((z * self.attn).sum(dim=-1))  # [num_edges]
        alpha = softmax(alpha, index)  # softmax over neighbors of each node
        alpha = self.dropout(alpha)
        return x_j * alpha.unsqueeze(-1)  # message: scaled neighbor feature

# Relational Graph Attention Network (inherits from RGCN, but uses RGATConv)
class RGAT(RGCN):
    def __init__(self, num_nodes, num_relations, embedding_dim, scoring='distmult', dropout=0.3):
        super().__init__(num_nodes, num_relations, embedding_dim, scoring=scoring, dropout=dropout)
        self.conv1 = RGATConv(embedding_dim, embedding_dim, num_relations, dropout=dropout)
        self.conv2 = RGATConv(embedding_dim, embedding_dim, num_relations, dropout=dropout)

    # Optionally override forward if you want to change the skip connection logic, otherwise you can omit it
    def forward(self, x, edge_index, edge_type):
        x = self.embedding(x)
        out1 = self.conv1(x, edge_index, edge_type)
        x = self.norm1(x + out1)
        x = F.relu(x)
        x = self.dropout(x)
        out2 = self.conv2(x, edge_index, edge_type)
        x = self.norm2(x + out2)
        x = F.relu(x)
        x = self.dropout(x)
        return x


# CompGCN layer with composition operator
class CompGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, op='mult', dropout=0.3):
        super().__init__(aggr='add')
        self.op = op
        self.rel_emb = nn.Embedding(num_relations * 2, in_channels)
        self.rel_linear = Linear(in_channels, out_channels)  # update relation embeddings
        self.node_linear = Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.rel_linear.weight)
        nn.init.xavier_uniform_(self.node_linear.weight)

    def compose(self, h_j, r):
        if self.op == 'mult':
            return h_j * r
        elif self.op == 'add':
            return h_j + r
        elif self.op == 'corr':
            return torch.fft.ifft(torch.fft.fft(h_j) * torch.conj(torch.fft.fft(r))).real
        else:
            raise ValueError(f"Unsupported op: {self.op}")

    def forward(self, x, edge_index, edge_type):
        rel = self.rel_emb(edge_type)
        self.updated_rels = self.rel_linear(self.rel_emb.weight)  # store updated relations
        out = self.propagate(edge_index, x=x, rel=rel)
        out = self.node_linear(out)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out, self.updated_rels

    def message(self, x_j, rel):
        return self.compose(x_j, rel)

# Composition-based Graph Convolutional Network
class CompGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, embedding_dim, scoring='distmult', dropout=0.3, op='mult'):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.rel_embedding = nn.Embedding(num_relations * 2, embedding_dim)
        self.conv1 = CompGCNLayer(embedding_dim, embedding_dim, num_relations, op=op, dropout=dropout)
        self.conv2 = CompGCNLayer(embedding_dim, embedding_dim, num_relations, op=op, dropout=dropout)
        self.scoring = scoring
        self.embedding_dim = embedding_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)

    def forward(self, x, edge_index, edge_type):
        x = self.embedding(x)
        x, _ = self.conv1(x, edge_index, edge_type)
        x, rels = self.conv2(x, edge_index, edge_type)
        return x, rels

    def get_score(self, h, r, t):
        if self.scoring == "distmult":
            return torch.sum(h * r * t, dim=-1)
        elif self.scoring == "transe":
            return -torch.norm(h + r - t, dim=-1)
        else:
            raise ValueError(f"Unsupported scoring: {self.scoring}")
