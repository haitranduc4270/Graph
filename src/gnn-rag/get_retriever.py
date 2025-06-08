import os
import json
import numpy as np
import pandas as pd
import torch
import faiss
from torch_geometric.utils import k_hop_subgraph
from rapidfuzz import process, fuzz

class EntityEmbeddingLoader:
    """
    Load and manage entity embeddings.
    """
    def __init__(self, entity_file: str):
        self.entity_file = entity_file
        self.embeddings = self._load_embeddings()
        self.index = self._build_faiss_index()

    def _load_embeddings(self) -> np.ndarray:
        """
        Load entity embeddings from a text file.
        Returns:
            np.ndarray: Embedding matrix [num_entities, dim]
        """
        embedding_list = []
        with open(self.entity_file, 'r') as f:
            for line in f:
                embedding_list.append([float(x) for x in line.strip().split()])
        return np.array(embedding_list, dtype=np.float32)

    def _build_faiss_index(self):
        """
        Build FAISS index from the loaded embeddings.
        Returns:
            faiss.Index: FAISS index object
        """
        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(self.embeddings)
        return index

    def save_index(self, path: str):
        faiss.write_index(self.index, path)

    def load_index(self, path: str):
        self.index = faiss.read_index(path)

    def search(self, query_embedding: np.ndarray, top_k: int = 1):
        """
        Search nearest entities in embedding space.

        Args:
            query_embedding (np.ndarray): shape [1, dim]
            top_k (int): number of neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]: distances and indices
        """
        return self.index.search(query_embedding, top_k)


class KnowledgeGraph:
    """
    Store knowledge graph structure and utilities.
    """
    def __init__(self, entities_path, relations_path, triples_path):
        self.entities_df = pd.read_csv(entities_path)
        self.relations_df = pd.read_csv(relations_path)
        self.triples = pd.read_csv(triples_path)
        self.entity_names = self.entities_df['entity'].astype(str).tolist()
        self.entity_id_map = self.entities_df.set_index('entity')['id'].to_dict()
        self.id_entity_map = self.entities_df.set_index('id')['entity'].to_dict()
        self.edge_index, self.relation_map = self._build_graph()

    def _build_graph(self):
        edge_list = []
        relation_map = {}

        for _, row in self.triples.iterrows():
            h, r, t = row['source'], row['relation'], row['target']
            h_id = self.entity_id_map.get(h)
            t_id = self.entity_id_map.get(t)
            if h_id is not None and t_id is not None:
                edge_list.append((h_id, t_id))
                edge_list.append((t_id, h_id))  # Undirected graph
                relation_map[(h_id, t_id)] = r
                relation_map[(t_id, h_id)] = r

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index, relation_map


class QARetriever:
    """
    Process QA data, retrieve subgraph, and save results.
    """
    def __init__(self, kg: KnowledgeGraph, embedder: EntityEmbeddingLoader, qa_file: str, output_csv: str, num_hops: int = 2):
        self.kg = kg
        self.embedder = embedder
        self.qa_file = qa_file
        self.output_csv = output_csv
        self.num_hops = num_hops

        with open(self.qa_file, "r", encoding="utf-8") as f:
            self.qa_data = json.load(f)

        if os.path.exists(self.output_csv):
            os.remove(self.output_csv)

    def _match_entities(self, mention: str):
        """
        Match a mention to the closest entity name using fuzzy matching.
        """
        return process.extractOne(mention, self.kg.entity_names, scorer=fuzz.WRatio)

    def _extract_subgraph(self, seed_ids):
        """
        Extract k-hop subgraph from the knowledge graph given seed nodes.
        """
        seed_ids_tensor = torch.tensor(list(seed_ids), dtype=torch.long)
        subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
            seed_ids_tensor, self.num_hops, self.kg.edge_index, relabel_nodes=False
        )
        return edge_index_sub

    def run(self):
        """
        Run the QA retrieval pipeline and save to CSV.
        """
        for idx, item in enumerate(self.qa_data):
            question = item.get("question", "")
            expected_answer = item.get("expected_answer", "")
            entities = item.get("entities", [])

            seed_ids = set()
            for ent in entities:
                match = self._match_entities(ent)
                if not match:
                    continue
                matched_name, score, matched_idx = match
                query_emb = self.embedder.embeddings[matched_idx].reshape(1, -1)
                _, I = self.embedder.search(query_emb, top_k=1)
                for i in I[0]:
                    seed_ids.add(i)

            if len(seed_ids) == 0:
                print(f"⚠️ Not enough nodes found for question: {question}")
                retriever_data = []
            else:
                edge_index_sub = self._extract_subgraph(seed_ids)
                retriever_data = []
                for i in range(edge_index_sub.shape[1]):
                    src = edge_index_sub[0, i].item()
                    dst = edge_index_sub[1, i].item()
                    rel = self.kg.relation_map.get((src, dst), "related_to")
                    src_name = self.kg.id_entity_map.get(src, f"node_{src}")
                    dst_name = self.kg.id_entity_map.get(dst, f"node_{dst}")
                    retriever_data.append(f"{src_name} {rel} {dst_name}")

            # Save result
            df_result = pd.DataFrame([{
                "query": question,
                "retriever_data": "\n".join(retriever_data),
                "expected_answer": expected_answer
            }])
            df_result.to_csv(self.output_csv, mode='a', index=False, header=(idx == 0))


# =========================
# Run the pipeline
# =========================

if __name__ == "__main__":
    kg = KnowledgeGraph(
        entities_path="../../data/entities.csv",
        relations_path="../../data/relations.csv",
        triples_path="../../data/triples.csv"
    )

    embedder = EntityEmbeddingLoader(entity_file="../../data/entity2vec.txt")
    embedder.save_index("faiss_index.index")  # Optional: Save index to disk
    embedder.load_index("faiss_index.index")  # Load from disk if needed

    retriever = QARetriever(
        kg=kg,
        embedder=embedder,
        qa_file="qa_with_entities.json",
        output_csv="qa_retriever_results.csv",
        num_hops=2
    )

    retriever.run()
