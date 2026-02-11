"""
Embedding Generation for Recommendation Systems

Multiple embedding strategies:
1. Matrix Factorization (collaborative filtering baseline)
2. Two-Tower Neural Network (user/item embeddings)
3. Transformer-based Sequential Embeddings (BERT4Rec/SASRec)
4. Multi-modal Embeddings (text + image)

Staff Interview Topics:
- Embedding dimensionality selection
- Cold-start problem handling
- Online serving of embeddings
- ANN (Approximate Nearest Neighbor) search
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    embedding_dim: int = 128
    user_vocab_size: int = 1000000
    item_vocab_size: int = 500000
    batch_size: int = 1024
    learning_rate: float = 0.001
    num_epochs: int = 10
    use_pretrained_text: bool = True
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class MatrixFactorization(nn.Module):
    """
    Classic Matrix Factorization (SVD-based)

    Interview Topic: When to use MF vs deep learning
    - MF: Simple, interpretable, good baseline
    - DL: Better for complex patterns, side information
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        use_bias: bool = True
    ):
        super().__init__()

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.user_bias = None

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence"""
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

        if self.user_bias is not None:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings/scores for user-item pairs

        Args:
            user_ids: [batch_size]
            item_ids: [batch_size]

        Returns:
            scores: [batch_size]
        """
        user_emb = self.user_embeddings(user_ids)  # [batch_size, embedding_dim]
        item_emb = self.item_embeddings(item_ids)  # [batch_size, embedding_dim]

        # Dot product
        scores = (user_emb * item_emb).sum(dim=1)  # [batch_size]

        # Add biases
        if self.user_bias is not None:
            scores = scores + self.user_bias(user_ids).squeeze() + \
                     self.item_bias(item_ids).squeeze() + self.global_bias

        return scores

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Retrieve user embedding for serving"""
        with torch.no_grad():
            emb = self.user_embeddings(torch.tensor([user_id])).numpy()
        return emb[0]

    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Retrieve item embedding for serving"""
        with torch.no_grad():
            emb = self.item_embeddings(torch.tensor([item_id])).numpy()
        return emb[0]


class TwoTowerModel(nn.Module):
    """
    Two-Tower Neural Network Architecture

    Interview Topic: Why two-tower architecture?
    - Separate user/item towers enable independent encoding
    - Can cache item embeddings (static)
    - Fast serving with ANN search
    - Easy to update user tower without recomputing items

    Key Design Decision: This is the industry standard for large-scale
    recommendation systems (YouTube, Google, Meta)
    """

    def __init__(
        self,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128]
    ):
        super().__init__()

        # User tower
        user_layers = []
        prev_dim = user_feature_dim
        for hidden_dim in hidden_dims:
            user_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        user_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)

        # Item tower
        item_layers = []
        prev_dim = item_feature_dim
        for hidden_dim in hidden_dims:
            item_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        item_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.item_tower = nn.Sequential(*item_layers)

        # Temperature parameter for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity scores

        Args:
            user_features: [batch_size, user_feature_dim]
            item_features: [batch_size, item_feature_dim]

        Returns:
            scores: [batch_size]
        """
        user_emb = self.user_tower(user_features)  # [batch_size, embedding_dim]
        item_emb = self.item_tower(item_features)  # [batch_size, embedding_dim]

        # L2 normalize (crucial for dot product similarity)
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        # Cosine similarity (dot product of normalized vectors)
        scores = (user_emb * item_emb).sum(dim=1) / self.temperature

        return scores

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user features to embedding"""
        user_emb = self.user_tower(user_features)
        return F.normalize(user_emb, p=2, dim=1)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features to embedding"""
        item_emb = self.item_tower(item_features)
        return F.normalize(item_emb, p=2, dim=1)


class SequentialRecommender(nn.Module):
    """
    Transformer-based Sequential Recommendation (SASRec-style)

    Interview Topic: When to use sequential models?
    - Captures temporal patterns
    - Session-based recommendations
    - Next-item prediction
    - Trade-off: More complex, higher latency

    Use cases: E-commerce, streaming, news
    """

    def __init__(
        self,
        item_vocab_size: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_length: int = 50,
        dropout: float = 0.2
    ):
        super().__init__()

        self.item_embedding = nn.Embedding(
            item_vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # Positional encoding
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_layer = nn.Linear(embedding_dim, item_vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.max_seq_length = max_seq_length

    def forward(
        self,
        item_sequences: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict next item given sequence

        Args:
            item_sequences: [batch_size, seq_length]
            padding_mask: [batch_size, seq_length] (True for padding)

        Returns:
            logits: [batch_size, seq_length, item_vocab_size]
        """
        batch_size, seq_length = item_sequences.shape

        # Item embeddings
        item_emb = self.item_embedding(item_sequences)  # [batch_size, seq_length, embedding_dim]

        # Positional embeddings
        positions = torch.arange(seq_length, device=item_sequences.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)  # [1, seq_length, embedding_dim]

        # Combine
        x = self.dropout(item_emb + pos_emb)

        # Causal mask (prevent looking ahead)
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device) * float('-inf'),
            diagonal=1
        )

        # Transformer encoding
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        # Project to item space
        logits = self.output_layer(x)

        return logits

    def get_sequence_embedding(
        self,
        item_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Get embedding representation of sequence (for user)

        Uses the last non-padding position
        """
        with torch.no_grad():
            item_emb = self.item_embedding(item_sequence)
            positions = torch.arange(len(item_sequence), device=item_sequence.device)
            pos_emb = self.position_embedding(positions)

            x = item_emb + pos_emb

            # Get last position embedding
            return x[-1]


class MultiModalEmbedding(nn.Module):
    """
    Multi-modal embedding combining text and image

    Interview Topic: Multi-modal learning in RecSys
    - Fusion strategies: early vs late fusion
    - Handling missing modalities
    - Computational trade-offs
    """

    def __init__(
        self,
        text_embedding_dim: int = 384,  # MiniLM output dim
        image_embedding_dim: int = 512,  # ResNet/CLIP output dim
        output_dim: int = 128,
        fusion_method: str = "concat"  # concat, attention, gated
    ):
        super().__init__()

        self.fusion_method = fusion_method

        if fusion_method == "concat":
            combined_dim = text_embedding_dim + image_embedding_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(combined_dim, output_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(output_dim * 2, output_dim)
            )

        elif fusion_method == "attention":
            # Cross-attention between modalities
            self.text_proj = nn.Linear(text_embedding_dim, output_dim)
            self.image_proj = nn.Linear(image_embedding_dim, output_dim)
            self.attention = nn.MultiheadAttention(output_dim, num_heads=4, batch_first=True)

        elif fusion_method == "gated":
            # Gated fusion (learnable weighting)
            self.text_proj = nn.Linear(text_embedding_dim, output_dim)
            self.image_proj = nn.Linear(image_embedding_dim, output_dim)
            self.gate = nn.Sequential(
                nn.Linear(text_embedding_dim + image_embedding_dim, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse text and image embeddings

        Args:
            text_emb: [batch_size, text_embedding_dim]
            image_emb: [batch_size, image_embedding_dim]

        Returns:
            fused_emb: [batch_size, output_dim]
        """
        if self.fusion_method == "concat":
            combined = torch.cat([text_emb, image_emb], dim=1)
            return self.fusion_layer(combined)

        elif self.fusion_method == "attention":
            text_proj = self.text_proj(text_emb).unsqueeze(1)  # [batch, 1, output_dim]
            image_proj = self.image_proj(image_emb).unsqueeze(1)  # [batch, 1, output_dim]

            # Cross-attention
            fused, _ = self.attention(text_proj, image_proj, image_proj)
            return fused.squeeze(1)

        elif self.fusion_method == "gated":
            text_proj = self.text_proj(text_emb)
            image_proj = self.image_proj(image_emb)

            gate_input = torch.cat([text_emb, image_emb], dim=1)
            gate_weight = self.gate(gate_input)

            return gate_weight * text_proj + (1 - gate_weight) * image_proj


class EmbeddingIndexer:
    """
    FAISS-based vector indexing for fast retrieval

    Interview Topic: ANN (Approximate Nearest Neighbor) search
    - FAISS vs ScaNN vs Annoy
    - Index types: Flat, IVF, HNSW
    - Recall vs latency trade-offs
    - Production considerations
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "IVF",
        nlist: int = 100,  # Number of clusters for IVF
        nprobe: int = 10   # Number of clusters to search
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = None

    def build_index(self, embeddings: np.ndarray):
        """
        Build FAISS index from embeddings

        Args:
            embeddings: [num_items, embedding_dim]

        Interview Points:
        - Flat: Exact search, O(n)
        - IVF: Cluster-based, O(k) where k << n
        - HNSW: Graph-based, best recall/speed trade-off
        """
        embeddings = embeddings.astype('float32')

        if self.index_type == "Flat":
            # Exact search (brute force)
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product

        elif self.index_type == "IVF":
            # Inverted file index (faster, approximate)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            # Train quantizer
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe

        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # M=32
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16

        # Add vectors
        self.index.add(embeddings)
        logger.info(f"Built {self.index_type} index with {self.index.ntotal} vectors")

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k nearest neighbors

        Args:
            query_embeddings: [num_queries, embedding_dim]
            k: Number of neighbors to return

        Returns:
            distances: [num_queries, k]
            indices: [num_queries, k]
        """
        query_embeddings = query_embeddings.astype('float32')
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices

    def save_index(self, path: str):
        """Save index to disk"""
        faiss.write_index(self.index, path)
        logger.info(f"Saved index to {path}")

    def load_index(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(path)
        logger.info(f"Loaded index from {path}")


# Training utilities
def train_matrix_factorization(
    model: MatrixFactorization,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cuda"
):
    """
    Train matrix factorization model

    Interview Topic: Loss functions for RecSys
    - Pointwise: MSE, BCE
    - Pairwise: BPR (Bayesian Personalized Ranking)
    - Listwise: ListMLE, ListNet
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            ratings = batch['rating'].to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)

            # Add L2 regularization manually if needed
            # l2_reg = sum(p.pow(2).sum() for p in model.parameters())
            # loss = loss + 1e-5 * l2_reg

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    # Example: Initialize and build index
    config = EmbeddingConfig(embedding_dim=128)

    # Create dummy embeddings
    num_items = 10000
    item_embeddings = np.random.randn(num_items, config.embedding_dim).astype('float32')

    # Build FAISS index
    indexer = EmbeddingIndexer(
        embedding_dim=config.embedding_dim,
        index_type="IVF"
    )
    indexer.build_index(item_embeddings)

    # Search
    query = np.random.randn(1, config.embedding_dim).astype('float32')
    distances, indices = indexer.search(query, k=10)
    print(f"Top 10 similar items: {indices[0]}")
