# Graph Neural Network Project

This project implements and experiments with Graph Neural Networks (GNNs) for graph data processing and analysis, particularly in the fields of Knowledge Graph and Question Answering. The project focuses on building a comprehensive system for knowledge graph construction, entity and relation embedding learning, and question answering using state-of-the-art GNN architectures.

## Key Features

### 1. Knowledge Graph Construction
- Automated data collection from product information
- Entity and relation extraction
- Knowledge graph construction using Neo4j
- Support for heterogeneous graph structures

### 2. GNN Models Implementation
- **RGCN (Relational Graph Convolutional Network)**
  - Multi-relational graph convolution
  - Basis decomposition for relation-specific transformations
  - Support for both directed and undirected graphs
  - Residual connections for better gradient flow

- **RGAT (Relational Graph Attention Network)**
  - Attention-based message passing
  - Relation-specific attention mechanisms
  - Multi-head attention support
  - Adaptive edge importance learning

- **CompGCN (Composition-based Graph Convolutional Network)**
  - Composition operations (multiplication, addition, circular correlation)
  - Joint entity and relation embedding learning
  - Support for inverse relations
  - Efficient message passing

- **ConvE (Convolutional 2D Knowledge Graph Embeddings)**
  - 2D convolutional neural networks for link prediction
  - Efficient parameter sharing
  - Support for complex relation patterns
  - State-of-the-art performance on standard benchmarks

### 3. Question Answering System
- GNN-based entity and relation understanding
- RAG (Retrieval-Augmented Generation) integration
- Support for complex queries
- Real-time response generation

### 4. Evaluation Framework
- Comprehensive evaluation metrics
- Link prediction evaluation
- Question answering accuracy
- Performance benchmarking

## Project Structure

```
.
├── data/               # Data directory
│   ├── entities.csv    # Entity list with IDs and types
│   ├── relations.csv   # Relation list with IDs and properties
│   └── triples.csv     # Knowledge graph triplets
├── data_prepare/       # Data preprocessing scripts
│   ├── data_crwaler.py # Product data crawler with rate limiting
│   └── kg_build.py     # Knowledge graph builder with Neo4j integration
├── gnn_models/         # GNN model implementations
│   ├── CompGCN.py      # CompGCN architectures and layers
│   ├── RGCN.py         # Variants of RGCN architectures and layers
│   ├── train.py        # Training pipeline with logging
│   └── utils.py        # Data processing and evaluation utilities
├── kg_embs/            # Implementation of knowledge graph embedding models
├── src/                # Main source code
│   ├── base_line/      # Baseline models and implementations
│   ├── gnn-rag/        # GNN-RAG implementation
│   └── evaluation.py   # Evaluation metrics and analysis
└── docker-compose.yaml # Docker configuration for deployment
```

## System Requirements

### Hardware Requirements

### Software Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- Neo4j Database 4.4+
- Docker 20.10+ (optional)

## Installation

1. Clone repository:
```bash
git clone [repository-url]
cd Graph
```

2. Create and activate virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install other dependencies
pip install -r requirements.txt
```

4. Configure Neo4j:
- Install Neo4j Desktop or Neo4j Community Edition
- Create a new database
- Update connection information in .env file:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

## Usage

### 1. Data Preparation

```bash
# Crawl product data
python data_prepare/data_crwaler.py

# Build knowledge graph
python data_prepare/kg_build.py
```

### 2. Training GNN Models

```bash
# Train RGAT model
python gnn_models/train.py \
    --data_dir data/ \
    --model rgat \
    --embedding_dim 100 \
    --dropout 0.2 \
    --lr 0.001 \
    --epochs 100 \
    --checkpoint_dir checkpoints/
```

Training parameters:
- `--model`: Model selection (rgat/rgcn/compgcn)
- `--embedding_dim`: Embedding dimension (default: 100)
- `--dropout`: Dropout rate (default: 0.2)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 64)
- `--num_bases`: Number of bases for RGCN (default: 30)
- `--checkpoint_dir`: Directory to save model checkpoints

### 3. Running Question Answering

```bash
# Run QA system
python src/base_line/ask_gemini_base_line.py
```

## Evaluation

### Link Prediction Metrics
- MRR (Mean Reciprocal Rank)
- Hits@1, Hits@3, Hits@10
- Area Under ROC Curve (AUC-ROC)
- Precision@K

### Question Answering Metrics
- ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- METEOR
- BLEU
- Exact Match

Run evaluation script:
```bash
python src/evaluation.py
```

## Model Performance

### Link Prediction Results
| Model   | MRR   | Hits@1 | Hits@3 | Hits@10 |
|---------|-------|--------|--------|---------|
| RGCN    | 0.XXX | 0.XXX  | 0.XXX  | 0.XXX   |
| RGAT    | 0.XXX | 0.XXX  | 0.XXX  | 0.XXX   |
| CompGCN | 0.XXX | 0.XXX  | 0.XXX  | 0.XXX   |
| ConvE   | 0.XXX | 0.XXX  | 0.XXX  | 0.XXX   |

### Question Answering Results
| Metric  | Score |
|---------|-------|
| ROUGE-1 | 0.XXX |
| ROUGE-2 | 0.XXX |
| ROUGE-L | 0.XXX |
| METEOR  | 0.XXX |

## Docker Deployment

Build and run with Docker:
```bash
# Build Docker image
docker build -t gnn-project .

# Run container
docker-compose up
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation
- Add type hints
- Include example usage

## License

This project is distributed under the MIT License. See the `LICENSE` file for more details.

## Citation

If you use this code in your research, please cite:
```
@misc{gnn-project,
  author = {},
  title = {Graph Neural Network Project},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/haitranduc4270/Graph}
}
```

## Contact

- Author: []
- Email: []
- GitHub: []
- LinkedIn: []