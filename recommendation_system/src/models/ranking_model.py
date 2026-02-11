"""
Ranking Models for Recommendation Systems

Stage 2 of Two-Stage Architecture: Re-rank candidate items

Models implemented:
1. LightGBM/XGBoost (tree-based, production workhorse)
2. Deep & Cross Network (DCN) - learns feature interactions
3. Wide & Deep - combines memorization + generalization
4. DeepFM - factorization machine + deep network

Staff Interview Topics:
- Tree-based vs neural ranking
- Feature interactions
- Calibration and serving
- Online learning
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RankingConfig:
    """Configuration for ranking models"""
    # LightGBM params
    num_trees: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 8
    num_leaves: int = 64
    min_data_in_leaf: int = 100

    # Neural network params
    embedding_dim: int = 32
    hidden_dims: List[int] = None
    dropout: float = 0.2
    batch_size: int = 2048

    # Training
    early_stopping_rounds: int = 50
    use_gpu: bool = True

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class LightGBMRanker:
    """
    LightGBM for Learning to Rank

    Interview Topic: Why tree-based models dominate RecSys ranking?
    - Handle mixed data types naturally
    - Robust to feature scaling
    - Built-in feature interactions
    - Fast training and inference
    - Interpretable (feature importance)

    When to use:
    - Have rich features (user, item, context)
    - Need interpretability
    - Limited training data (< 1B samples)
    - Low latency requirement (< 10ms)
    """

    def __init__(self, config: RankingConfig):
        self.config = config
        self.model = None
        self.feature_names = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        query_groups_train: Optional[np.ndarray] = None,
        query_groups_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Train LightGBM ranking model

        Args:
            X_train: [num_samples, num_features]
            y_train: [num_samples] - labels (1 for positive, 0 for negative)
            query_groups: Group sizes for ranking (optional)
                For example: [10, 15, 20] means first 10 samples are one query,
                next 15 are another query, etc.

        Interview Point: LambdaRank objective
        - Optimizes NDCG directly (better than binary classification)
        - Pairwise ranking approach
        """
        self.feature_names = feature_names

        # Create datasets
        if query_groups_train is not None:
            # Learning to Rank mode
            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                group=query_groups_train,
                feature_name=feature_names
            )
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                group=query_groups_val,
                feature_name=feature_names,
                reference=train_data
            )
            objective = 'lambdarank'
            metric = 'ndcg'
        else:
            # Binary classification mode (simpler, but less optimal)
            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                feature_name=feature_names
            )
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                feature_name=feature_names,
                reference=train_data
            )
            objective = 'binary'
            metric = 'auc'

        params = {
            'objective': objective,
            'metric': metric,
            'boosting_type': 'gbdt',
            'num_leaves': self.config.num_leaves,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'min_data_in_leaf': self.config.min_data_in_leaf,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1,
            'device': 'gpu' if self.config.use_gpu else 'cpu'
        }

        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.num_trees,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds),
                lgb.log_evaluation(50)
            ]
        )

        logger.info("LightGBM training complete!")

        # Feature importance analysis
        self.log_feature_importance()

        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict scores for ranking"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)

    def log_feature_importance(self, top_k: int = 20):
        """
        Log top-k important features

        Interview Topic: Feature importance in production
        - Helps debug model behavior
        - Identify data issues
        - Guide feature engineering
        """
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names or [f'f{i}' for i in range(len(importance))],
            'importance': importance
        }).sort_values('importance', ascending=False)

        logger.info(f"\nTop {top_k} important features:")
        logger.info(feature_importance.head(top_k).to_string())

    def save_model(self, path: str):
        """Save model to disk"""
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk"""
        self.model = lgb.Booster(model_file=path)
        logger.info(f"Model loaded from {path}")


class DeepCrossNetwork(nn.Module):
    """
    Deep & Cross Network (DCN)

    Interview Topic: Automatic feature crossing
    - Cross Network: learns bounded-degree feature interactions explicitly
    - Deep Network: learns arbitrary interactions implicitly
    - Best of both worlds

    Paper: "Deep & Cross Network for Ad Click Predictions" (Google, 2017)

    When to use:
    - Large-scale data (> 1B samples)
    - Need automatic feature interactions
    - Can afford higher latency (20-50ms)
    """

    def __init__(
        self,
        num_features: int,
        embedding_dims: Dict[str, int],  # {feature_name: embedding_dim}
        vocab_sizes: Dict[str, int],      # {feature_name: vocab_size}
        cross_layers: int = 3,
        deep_layers: List[int] = [512, 256, 128],
        dropout: float = 0.2
    ):
        super().__init__()

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, emb_dim)
            for name, (vocab_size, emb_dim) in
            zip(vocab_sizes.keys(), zip(vocab_sizes.values(), embedding_dims.values()))
        })

        # Calculate total input dimension
        self.total_emb_dim = sum(embedding_dims.values())
        self.input_dim = num_features + self.total_emb_dim

        # Cross Network
        self.cross_layers = nn.ModuleList([
            CrossLayer(self.input_dim) for _ in range(cross_layers)
        ])

        # Deep Network
        deep_network = []
        prev_dim = self.input_dim
        for hidden_dim in deep_layers:
            deep_network.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.deep_network = nn.Sequential(*deep_network)

        # Final combination layer
        self.final_layer = nn.Linear(self.input_dim + deep_layers[-1], 1)

    def forward(
        self,
        numeric_features: torch.Tensor,
        categorical_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            numeric_features: [batch_size, num_numeric_features]
            categorical_features: {feature_name: [batch_size]}

        Returns:
            scores: [batch_size, 1]
        """
        # Embed categorical features
        embeddings = []
        for name, indices in categorical_features.items():
            emb = self.embeddings[name](indices)
            embeddings.append(emb)

        # Concatenate all features
        if embeddings:
            cat_emb = torch.cat(embeddings, dim=1)
            x = torch.cat([numeric_features, cat_emb], dim=1)
        else:
            x = numeric_features

        # Cross Network
        x_cross = x
        for cross_layer in self.cross_layers:
            x_cross = cross_layer(x_cross, x)

        # Deep Network
        x_deep = self.deep_network(x)

        # Concatenate and predict
        combined = torch.cat([x_cross, x_deep], dim=1)
        output = self.final_layer(combined)

        return output.squeeze(1)


class CrossLayer(nn.Module):
    """
    Single cross layer for DCN

    Computes: x_l+1 = x_0 * x_l^T * w_l + b_l + x_l

    Interview Point: How does this learn interactions?
    - Explicit element-wise multiplication with input x_0
    - Bounded degree (controlled by number of layers)
    - Efficient: O(d) parameters instead of O(d^2)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1))
        self.bias = nn.Parameter(torch.zeros(input_dim))

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] - current layer input
            x0: [batch_size, input_dim] - original input (x_0)

        Returns:
            [batch_size, input_dim]
        """
        # x_l^T * w_l
        xw = torch.matmul(x, self.weight)  # [batch_size, 1]

        # x_0 * (x_l^T * w_l)
        x0_xw = x0 * xw  # [batch_size, input_dim]

        # + b_l + x_l
        return x0_xw + self.bias + x


class DeepFM(nn.Module):
    """
    DeepFM: Factorization Machine + Deep Neural Network

    Interview Topic: FM component
    - Captures 2nd order feature interactions efficiently
    - Especially good for sparse features
    - O(kn) instead of O(n^2) for interactions

    Paper: "DeepFM: A Factorization-Machine based Neural Network" (Huawei, 2017)

    When to use:
    - Sparse categorical features dominate
    - Need both low and high order interactions
    - Click prediction, conversion prediction
    """

    def __init__(
        self,
        num_numeric_features: int,
        vocab_sizes: Dict[str, int],
        embedding_dim: int = 16,
        deep_layers: List[int] = [256, 128, 64],
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_numeric = num_numeric_features
        self.embedding_dim = embedding_dim

        # Embeddings for categorical features (used in both FM and Deep)
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embedding_dim)
            for name, vocab_size in vocab_sizes.items()
        })

        # Linear part (1st order)
        total_features = num_numeric_features + len(vocab_sizes)
        self.linear = nn.Linear(total_features, 1)

        # Deep part
        deep_input_dim = num_numeric_features + len(vocab_sizes) * embedding_dim
        deep_network = []
        prev_dim = deep_input_dim
        for hidden_dim in deep_layers:
            deep_network.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        deep_network.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*deep_network)

    def forward(
        self,
        numeric_features: torch.Tensor,
        categorical_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass combining FM and Deep components

        Args:
            numeric_features: [batch_size, num_numeric_features]
            categorical_features: {feature_name: [batch_size]}

        Returns:
            scores: [batch_size]
        """
        batch_size = numeric_features.shape[0]

        # Get embeddings
        embeddings = []
        linear_parts = []

        for name, indices in categorical_features.items():
            emb = self.embeddings[name](indices)  # [batch_size, embedding_dim]
            embeddings.append(emb)

            # For linear part, use sum of embedding
            linear_parts.append(emb.sum(dim=1, keepdim=True))

        # Stack embeddings for FM
        if embeddings:
            emb_stack = torch.stack(embeddings, dim=1)  # [batch_size, num_fields, emb_dim]

        # 1. Linear (1st order) component
        linear_features = torch.cat([numeric_features] + linear_parts, dim=1)
        linear_score = self.linear(linear_features)

        # 2. FM (2nd order) component
        # sum of squares
        sum_of_squares = torch.sum(emb_stack, dim=1) ** 2  # [batch_size, emb_dim]

        # square of sums
        square_of_sums = torch.sum(emb_stack ** 2, dim=1)  # [batch_size, emb_dim]

        # FM interaction: 0.5 * (sum_of_squares - square_of_sums)
        fm_score = 0.5 * torch.sum(sum_of_squares - square_of_sums, dim=1, keepdim=True)

        # 3. Deep component
        deep_input = torch.cat([numeric_features] + embeddings, dim=1)
        deep_score = self.deep(deep_input)

        # Combine all components
        total_score = linear_score + fm_score + deep_score

        return total_score.squeeze(1)


class RankingMetrics:
    """
    Evaluation metrics for ranking models

    Interview Topic: What metrics matter for RecSys?
    """

    @staticmethod
    def compute_ndcg(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Normalized Discounted Cumulative Gain

        Interview Point: Why NDCG > Accuracy?
        - Position matters (top items more important)
        - Handles graded relevance (not just binary)
        - Industry standard metric
        """
        return ndcg_score([y_true], [y_pred], k=k)

    @staticmethod
    def compute_map_at_k(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 10
    ) -> float:
        """Mean Average Precision @ K"""
        order = np.argsort(y_pred)[::-1][:k]
        y_true_sorted = y_true[order]

        if y_true_sorted.sum() == 0:
            return 0.0

        precisions = []
        num_relevant = 0
        for i, relevant in enumerate(y_true_sorted):
            if relevant:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))

        return np.mean(precisions) if precisions else 0.0

    @staticmethod
    def compute_mrr(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Mean Reciprocal Rank"""
        order = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[order]

        for i, relevant in enumerate(y_true_sorted):
            if relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def evaluate_ranking(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """Comprehensive ranking evaluation"""
        metrics = {
            'auc': roc_auc_score(y_true, y_pred),
            'logloss': log_loss(y_true, y_pred),
            'mrr': RankingMetrics.compute_mrr(y_true, y_pred)
        }

        for k in k_values:
            metrics[f'ndcg@{k}'] = RankingMetrics.compute_ndcg(y_true, y_pred, k)
            metrics[f'map@{k}'] = RankingMetrics.compute_map_at_k(y_true, y_pred, k)

        return metrics


# Example usage
if __name__ == "__main__":
    # Example: Train LightGBM ranker
    config = RankingConfig(
        num_trees=500,
        learning_rate=0.05,
        use_gpu=False
    )

    ranker = LightGBMRanker(config)

    # Generate dummy data
    X_train = np.random.randn(10000, 50)
    y_train = np.random.randint(0, 2, 10000)
    X_val = np.random.randn(2000, 50)
    y_val = np.random.randint(0, 2, 2000)

    feature_names = [f'feature_{i}' for i in range(50)]

    # Train
    ranker.train(X_train, y_train, X_val, y_val, feature_names=feature_names)

    # Predict
    predictions = ranker.predict(X_val)

    # Evaluate
    metrics = RankingMetrics.evaluate_ranking(y_val, predictions)
    print(f"Evaluation metrics: {metrics}")
