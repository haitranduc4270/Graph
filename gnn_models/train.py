import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from models import RGCN, RGAT, CompGCN
from utils import load_entities_relations, read_triplets, build_data, get_true_tails, evaluate

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--model', type=str, default='rgat', choices=['rgat', 'rgcn', 'compgcn'], help='Model type to use')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of entity and relation embeddings')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save model checkpoints')
    args = parser.parse_args()

    # Load data and prepare model
    data_dir = args.data_dir
    entity2id, relation2id = load_entities_relations(data_dir)
    train_sup = read_triplets(os.path.join(data_dir, 'train.csv'), entity2id, relation2id)
    val_sup = read_triplets(os.path.join(data_dir, 'valid.csv'), entity2id, relation2id)
    test_sup = read_triplets(os.path.join(data_dir, 'test.csv'), entity2id, relation2id)
    num_entities = len(entity2id)
    num_relations = len(relation2id)
    data = build_data(train_sup, num_entities)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model selection
    if args.model == 'rgat':
        model = RGAT(num_entities, num_relations, embedding_dim=args.embedding_dim, dropout=args.dropout).to(device)
    elif args.model == 'rgcn':
        model = RGCN(num_entities, num_relations, embedding_dim=args.embedding_dim, dropout=args.dropout).to(device)
    elif args.model == 'compgcn':
        model = CompGCN(num_entities, num_relations, embedding_dim=args.embedding_dim, dropout=args.dropout).to(device)
    else:
        raise ValueError("Unknown model type")

    print(f"Using model: {args.model}, Embedding Dim: {args.embedding_dim}, Dropout: {args.dropout}")
    print(f"Number of entities: {num_entities}, Number of relations: {num_relations}")

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    margin_criterion = nn.MarginRankingLoss(margin=1.0)
    bce_criterion = nn.BCEWithLogitsLoss()
    true_tails = get_true_tails(train_sup + val_sup + test_sup)
    
    # Training loop
    checkpoints = args.checkpoint_dir
    os.makedirs(checkpoints, exist_ok=True)
    best_mrr = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        x = torch.arange(num_entities).to(device)

        # Forward pass
        node_emb = model(x, data.edge_index.to(device), data.edge_type.to(device))
        heads = torch.tensor([h for h, _, _ in train_sup], device=device)
        rels = torch.tensor([r for _, r, _ in train_sup], device=device)
        tails = torch.tensor([t for _, _, t in train_sup], device=device)

        # Negative sampling
        neg_tails = torch.randint(0, num_entities, (len(tails),), device=device)
        h = node_emb[heads]
        r = model.rel_embedding.weight(rels)
        t = node_emb[tails]
        t_neg = node_emb[neg_tails]

        # Compute scores and loss
        pos_score = model.get_score(h, r, t)
        neg_score = model.get_score(h, r, t_neg)
        
        if model.scoring == 'distmult':
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
            loss = bce_criterion(scores, labels)
        else:
            target = torch.ones_like(pos_score)
            loss = margin_criterion(pos_score, neg_score, target)
        loss.backward()
        optimizer.step()

        # Evaluate on validation set
        mrr, recall = evaluate(model, val_sup, data, num_entities, true_tails, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Val MRR: {mrr:.4f}, Recall@10: {recall:.4f}")
        if mrr > best_mrr:
            best_mrr = mrr
            print(f"New best MRR: {best_mrr:.4f}")
            torch.save(model.state_dict(), os.path.join(checkpoints, f"{args.model}.pt"))
            print("Model saved\n")

    # Final evaluation on test set
    mrr, recall = evaluate(model, test_sup, data, num_entities, true_tails, device)
    print(f"Test MRR: {mrr:.4f}, Recall@10: {recall:.4f}")

if __name__ == "__main__":
    main()
