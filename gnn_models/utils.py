import os
import pandas as pd
import torch
from collections import defaultdict

# Load entity and relation mappings from CSV files
def load_entities_relations(data_path):
    entities_df = pd.read_csv(os.path.join(data_path, 'entities.csv'))
    relations_df = pd.read_csv(os.path.join(data_path, 'relations.csv'))
    entity2id = dict(zip(entities_df['entity'], entities_df['id'].astype(int)))
    relation2id = dict(zip(relations_df['relation'], relations_df['id'].astype(int)))
    return entity2id, relation2id

# Read triplets from CSV and convert to id tuples
def read_triplets(file_path, entity2id, relation2id):
    triplets = []
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        head, relation, tail = row[0], row[1], row[2]
        triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))
    return triplets

# Build PyG Data object from triplets
def build_data(train_sup, num_entities):
    import torch
    from torch_geometric.data import Data
    edges = [(h, t, r) for (h, r, t) in train_sup]
    edge_pairs = [(h, t) for h, t, r in edges] + [(t, h) for h, t, r in edges]
    edge_types = [r for _, _, r in edges] * 2
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    x = torch.arange(num_entities)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)

# Build dictionary of true tails for filtered ranking
def get_true_tails(triplets_list):
    true_tails = defaultdict(set)
    for h, r, t in triplets_list:
        true_tails[(h, r)].add(t)
    return true_tails

# Evaluate model using MRR and Recall@10
def evaluate(model, eval_triples, data, num_entities, true_tails, device):
    model.eval()
    with torch.no_grad():
        x = torch.arange(num_entities).to(device)
        node_emb = model(x, data.edge_index.to(device), data.edge_type.to(device))
        rel_emb = model.rel_embedding.weight
        ranks = []
        for h, r, t in eval_triples:
            h_emb = node_emb[h]
            r_emb = rel_emb[r]
            scores = model.get_score(h_emb, r_emb, node_emb)
            filt = true_tails[(h, r)] - {t}
            if filt:
                scores[list(filt)] = float('-inf')
            sorted_idx = torch.argsort(scores, descending=True)
            rank = (sorted_idx == t).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
        mrr = torch.mean(1.0 / torch.tensor(ranks, dtype=torch.float)).item()
        recall_at_10 = (torch.tensor(ranks) <= 10).sum().item() / len(ranks)
        return mrr, recall_at_10
