import torch
from collections import defaultdict

def smooth_labels(labels, epsilon=0.1):
    # labels: tensor of 0s and 1s
    return labels * (1 - epsilon) + (1 - labels) * epsilon

def evaluate_conve(model, val_triples, all_triples, num_entities, device='cuda'):
    model.eval()
    with torch.no_grad():
        known_triples = set(all_triples)
        hits10 = 0
        mrr = 0
        count = 0
        for h, r, t in val_triples:
            h_tensor = torch.tensor([h], dtype=torch.long, device=device)
            r_tensor = torch.tensor([r], dtype=torch.long, device=device)
            logits = model(h_tensor, r_tensor).squeeze(0)
            mask = torch.zeros(num_entities, dtype=torch.bool, device=device)
            for corrupt_t in range(num_entities):
                if corrupt_t != t and (h, r, corrupt_t) in known_triples:
                    mask[corrupt_t] = True
            logits[mask] = -1e6
            _, indices = torch.sort(logits, descending=True)
            rank = (indices == t).nonzero(as_tuple=True)[0].item() + 1
            mrr += 1.0 / rank
            if rank <= 10:
                hits10 += 1
            count += 1
        return mrr / count, hits10 / count
    
def evaluate_conve_batch(model, eval_loader, all_triples, num_entities, device='cuda', batch_size=128):
    model.eval()
    with torch.no_grad():
        known_triples = set(all_triples)
        hits10 = 0
        mrr = 0
        count = 0

        for batch in eval_loader:
            h = batch[0].to(device)
            r = batch[1].to(device)
            t = batch[2].to(device)
            B = h.size(0)

            logits = model(h, r)  # [B, num_entities]

            # Mask known true tails except the target t for each sample in batch
            mask = torch.zeros((B, num_entities), dtype=torch.bool, device=device)
            for i in range(B):
                h_i, r_i, t_i = h[i].item(), r[i].item(), t[i].item()
                for corrupt_t in range(num_entities):
                    if corrupt_t != t_i and (h_i, r_i, corrupt_t) in known_triples:
                        mask[i, corrupt_t] = True
            logits = logits.masked_fill(mask, -1e6)

            # Compute ranks and hits@10
            sorted_indices = torch.argsort(logits, dim=1, descending=True)
            t = t.view(-1, 1)
            
            for idx in range(B):
                rank = (sorted_indices[idx] == t[idx, 0]).nonzero(as_tuple=True)[0].item() + 1
                mrr += 1.0 / rank
                if rank <= 10:
                    hits10 += 1
                count += 1

        return mrr / count, hits10 / count

def sample_negatives(batch_size, num_negatives, num_entities, true_tails_list):
    neg_samples = []
    for true_tails in true_tails_list:
        candidates = torch.randint(0, num_entities, (num_negatives * 2,))
        mask = ~torch.isin(candidates, torch.tensor(true_tails))
        filtered = candidates[mask][:num_negatives]
        while len(filtered) < num_negatives:
            extra = torch.randint(0, num_entities, (num_negatives,))
            mask = ~torch.isin(extra, torch.tensor(true_tails))
            filtered = torch.cat([filtered, extra[mask]])[:num_negatives]
        neg_samples.append(filtered)
    return torch.stack(neg_samples)
