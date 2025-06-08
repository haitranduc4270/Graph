import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=200, embedding_shape=(10, 20), dropout=0.3):
        super(ConvE, self).__init__()
        self.embedding_dim = embedding_dim
        self.emb_shape1, self.emb_shape2 = embedding_shape
        assert self.emb_shape1 * self.emb_shape2 == embedding_dim, "embedding_shape must multiply to embedding_dim"
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.conv2d = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.fc = nn.Linear(32 * self.emb_shape1 * 2 * self.emb_shape2, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.feature_dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(dropout)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_entities)))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, head_idx, relation_idx):
        head = self.entity_embedding(head_idx)
        rel = self.relation_embedding(relation_idx)
        B = head.size(0)
        head = head.view(-1, 1, self.emb_shape1, self.emb_shape2)
        rel = rel.view(-1, 1, self.emb_shape1, self.emb_shape2)
        x = torch.cat([head, rel], 2)
        x = self.bn0(x)
        x = self.input_dropout(x)
        x = self.conv2d(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_dropout(x)
        x = x.view(B, -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.matmul(x, self.entity_embedding.weight.transpose(1, 0))
        x += self.bias
        return x

    def score_triple(self, head_idx, relation_idx, tail_idx):
        logits = self.forward(head_idx, relation_idx)
        return logits.gather(1, tail_idx.view(-1, 1)).squeeze()

    def score_triples(self, head_idx, relation_idx, tail_idx):
        head_emb = self.entity_embedding(head_idx)
        rel_emb = self.relation_embedding(relation_idx)
        tail_emb = self.entity_embedding(tail_idx)
        return torch.sum(head_emb * rel_emb * tail_emb, dim=1)

