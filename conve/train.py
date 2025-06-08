import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from graphml_project.conve.conve import ConvE
from graphml_project.conve.utils import evaluate_conve, smooth_labels, sample_negatives
import torch.nn.functional as F
import argparse

class TripleDataset(Dataset):
    def __init__(self, triples, num_entities):
        self.triples = triples
        self.num_entities = num_entities
        self.triple_dict = self.build_triple_dict(triples)

    def build_triple_dict(self, triples):
        triple_dict = {}
        for h, r, t in triples:
            if (h, r) not in triple_dict:
                triple_dict[(h, r)] = []
            triple_dict[(h, r)].append(t)
        return triple_dict

    def __len__(self):
        return len(self.triple_dict)

    def __getitem__(self, idx):
        hr_pairs = list(self.triple_dict.keys())
        h, r = hr_pairs[idx]
        t_list = self.triple_dict[(h, r)]
        label = torch.zeros(self.num_entities)
        label[t_list] = 1.0
        return torch.tensor(h), torch.tensor(r), label

def load_data(data_dir):
    entities_df = pd.read_csv(os.path.join(data_dir, 'entities.csv'))
    relations_df = pd.read_csv(os.path.join(data_dir, 'relations.csv'))
    entity2id = dict(zip(entities_df['entity'], entities_df['id'].astype(int)))
    relation2id = dict(zip(relations_df['relation'], relations_df['id'].astype(int)))
    def read_triplets(file_path, entity2id, relation2id):
        triplets = []
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            head, relation, tail = row[0], row[1], row[2]
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))
        return triplets
    train_sup = read_triplets(os.path.join(data_dir, 'train.csv'), entity2id, relation2id)
    val_sup = read_triplets(os.path.join(data_dir, 'valid.csv'), entity2id, relation2id)
    test_sup = read_triplets(os.path.join(data_dir, 'test.csv'), entity2id, relation2id)
    return entity2id, relation2id, train_sup, val_sup, test_sup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--embedding_dim', type=int, default=200, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_negatives', type=int, default=20, help='Number of negative samples per positive')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for self-adversarial negative sampling')
    parser.add_argument('--use_self_adversarial', action='store_true', help='Enable self-adversarial negative sampling')
    parser.add_argument('--checkpoint', type=str, default='best_conve_model.pt', help='Path to save best model')
    args = parser.parse_args()

    data_dir = args.data_dir
    entity2id, relation2id, train_sup, val_sup, test_sup = load_data(data_dir)
    num_entities = len(entity2id)
    num_relations = len(relation2id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    dataset = TripleDataset(train_sup, num_entities)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model selection
    model = ConvE(num_entities, num_relations, embedding_dim=args.embedding_dim, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    bce_criterion = nn.BCEWithLogitsLoss()
    best_mrr = 0
    all_triples_set = set(train_sup + val_sup + test_sup)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for h, r, label in train_loader:
            h, r, label = h.to(device), r.to(device), label.to(device)

            if args.use_self_adversarial:
                batch_size = h.size(0)
                num_entities = label.size(1)
                # Extract true tails indices per triple
                true_tails_list = [label[i].nonzero(as_tuple=False).view(-1).tolist() for i in range(batch_size)]
                # Sample negative tails avoiding true tails
                neg_tails = sample_negatives(batch_size, args.num_negatives, num_entities, true_tails_list).to(device)  # [B, num_neg]
                # Compute positive logits for all entities (including true tails)
                logits = model(h, r)  # [B, num_entities]
                # Get negative logits via advanced indexing
                batch_indices = torch.arange(batch_size).unsqueeze(1).to(device)  # [B, 1]
                neg_logits = logits[batch_indices, neg_tails]  # [B, num_neg]
                # Self-adversarial weights on negative logits (softmax with temperature)
                neg_weights = F.softmax(neg_logits * args.temperature, dim=1).detach()  # [B, num_neg]
                # Positive loss: BCE with full label
                pos_loss = F.binary_cross_entropy_with_logits(logits, label, reduction='none').sum(dim=1).mean()
                # Negative loss: weighted BCE with zero labels for negatives
                neg_loss = - (neg_weights * F.logsigmoid(-neg_logits)).sum(dim=1).mean()
                loss = pos_loss + neg_loss
            else:
                logits = model(h, r)
                loss = bce_criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_mrr, val_hits10 = evaluate_conve(model, val_sup, all_triples_set, num_entities, device)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val MRR={val_mrr:.4f}, Hits@10={val_hits10:.4f}")
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            torch.save(model.state_dict(), args.checkpoint)
            print("New best model saved.")
    print(f"\nBest Val MRR: {best_mrr:.4f}")
    # Test evaluation
    mrr, hits10 = evaluate_conve(model, test_sup, all_triples_set, num_entities, device)
    print(f"Test MRR: {mrr:.4f}, Hits@10: {hits10:.4f}")

if __name__ == "__main__":
    main()
