#!/usr/bin/env python3
"""
Demo: End-to-end Recommendation Pipeline

This script demonstrates the complete pipeline with synthetic data.
Perfect for understanding the flow and testing components.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn

print("=" * 80)
print("PRODUCTION-GRADE RECOMMENDATION SYSTEM PIPELINE DEMO")
print("=" * 80)

# ============================================================================
# 1. SYNTHETIC DATA GENERATION
# ============================================================================
print("\n[1/7] Generating Synthetic Data...")

np.random.seed(42)

# Simulate user-item interactions
num_users = 10000
num_items = 5000
num_interactions = 100000

user_ids = np.random.randint(0, num_users, num_interactions)
item_ids = np.random.randint(0, num_items, num_interactions)
timestamps = [
    datetime.now() - timedelta(days=np.random.randint(0, 90))
    for _ in range(num_interactions)
]
event_types = np.random.choice(
    ['view', 'click', 'add_to_cart', 'purchase'],
    num_interactions,
    p=[0.6, 0.25, 0.1, 0.05]
)

interactions_df = pd.DataFrame({
    'user_id': user_ids,
    'item_id': item_ids,
    'timestamp': timestamps,
    'event_type': event_types,
    'rating': np.random.randint(1, 6, num_interactions)  # 1-5 stars
})

print(f"✓ Generated {len(interactions_df):,} interactions")
print(f"  - {num_users:,} users")
print(f"  - {num_items:,} items")
print(f"\nEvent distribution:")
print(interactions_df['event_type'].value_counts())

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2/7] Feature Engineering...")

# User features: aggregate statistics
user_features = interactions_df.groupby('user_id').agg({
    'item_id': 'count',  # Total interactions
    'timestamp': ['min', 'max'],  # First and last interaction
    'rating': 'mean'  # Average rating given
}).reset_index()

user_features.columns = ['user_id', 'total_interactions', 'first_seen', 'last_seen', 'avg_rating']
user_features['recency_days'] = (datetime.now() - user_features['last_seen']).dt.days
user_features['tenure_days'] = (user_features['last_seen'] - user_features['first_seen']).dt.days

print(f"✓ Created user features: {user_features.shape}")
print(f"  Feature columns: {list(user_features.columns)}")

# Item features: popularity metrics
item_features = interactions_df.groupby('item_id').agg({
    'user_id': 'count',  # Total views
    'rating': 'mean'  # Average rating
}).reset_index()

item_features.columns = ['item_id', 'popularity', 'avg_rating']
item_features['log_popularity'] = np.log1p(item_features['popularity'])

print(f"✓ Created item features: {item_features.shape}")
print(f"  Feature columns: {list(item_features.columns)}")

# ============================================================================
# 3. TRAIN/TEST SPLIT (Time-based)
# ============================================================================
print("\n[3/7] Creating Train/Test Split...")

# Time-based split: last 7 days for test
cutoff_date = datetime.now() - timedelta(days=7)

train_df = interactions_df[interactions_df['timestamp'] < cutoff_date]
test_df = interactions_df[interactions_df['timestamp'] >= cutoff_date]

print(f"✓ Train set: {len(train_df):,} interactions")
print(f"✓ Test set: {len(test_df):,} interactions")

# ============================================================================
# 4. MATRIX FACTORIZATION MODEL
# ============================================================================
print("\n[4/7] Training Matrix Factorization Model...")


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        # Dot product
        scores = (user_emb * item_emb).sum(dim=1)

        # Add biases
        scores += self.user_bias(user_ids).squeeze() + \
                  self.item_bias(item_ids).squeeze() + \
                  self.global_bias

        return scores


# Prepare data
train_users = torch.LongTensor(train_df['user_id'].values)
train_items = torch.LongTensor(train_df['item_id'].values)
train_ratings = torch.FloatTensor(train_df['rating'].values)

# Create model
model = MatrixFactorization(num_users, num_items, embedding_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Train (simplified - just a few batches for demo)
batch_size = 1024
num_batches = min(10, len(train_df) // batch_size)

model.train()
losses = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_users = train_users[start_idx:end_idx]
    batch_items = train_items[start_idx:end_idx]
    batch_ratings = train_ratings[start_idx:end_idx]

    optimizer.zero_grad()
    predictions = model(batch_users, batch_items)
    loss = criterion(predictions, batch_ratings)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

print(f"✓ Trained for {num_batches} batches")
print(f"  Final loss: {losses[-1]:.4f}")

# ============================================================================
# 5. GENERATE EMBEDDINGS
# ============================================================================
print("\n[5/7] Generating User and Item Embeddings...")

model.eval()
with torch.no_grad():
    # Get all item embeddings
    all_item_ids = torch.arange(num_items)
    item_embeddings = model.item_embeddings(all_item_ids).numpy()

    # Sample user embeddings
    sample_user_ids = torch.arange(min(1000, num_users))
    user_embeddings = model.user_embeddings(sample_user_ids).numpy()

print(f"✓ Generated embeddings:")
print(f"  - Item embeddings: {item_embeddings.shape}")
print(f"  - User embeddings: {user_embeddings.shape}")

# ============================================================================
# 6. CANDIDATE GENERATION (ANN Search Simulation)
# ============================================================================
print("\n[6/7] Simulating Candidate Generation...")

# For demo, use simple numpy similarity (in production, use FAISS)
def get_top_k_similar_items(user_emb, item_embs, k=50):
    """Find top-k similar items using cosine similarity"""
    # Normalize
    user_emb_norm = user_emb / np.linalg.norm(user_emb)
    item_embs_norm = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    # Compute similarities
    similarities = item_embs_norm @ user_emb_norm

    # Get top-k
    top_k_indices = np.argsort(-similarities)[:k]
    top_k_scores = similarities[top_k_indices]

    return top_k_indices, top_k_scores


# Get recommendations for first user
test_user_id = 0
test_user_emb = user_embeddings[test_user_id]

candidate_items, candidate_scores = get_top_k_similar_items(
    test_user_emb,
    item_embeddings,
    k=50
)

print(f"✓ Generated {len(candidate_items)} candidates for user {test_user_id}")
print(f"  Top 5 items: {candidate_items[:5]}")
print(f"  Top 5 scores: {candidate_scores[:5].round(3)}")

# ============================================================================
# 7. EVALUATION METRICS
# ============================================================================
print("\n[7/7] Computing Evaluation Metrics...")


def compute_metrics(predictions, actuals, k=10):
    """Compute recommendation metrics"""
    # Precision@K
    relevant = set(actuals)
    recommended = set(predictions[:k])
    precision = len(relevant & recommended) / k if k > 0 else 0

    # Recall@K
    recall = len(relevant & recommended) / len(relevant) if len(relevant) > 0 else 0

    # NDCG@K (simplified)
    dcg = sum([1.0 / np.log2(i + 2) if predictions[i] in relevant else 0
               for i in range(min(k, len(predictions)))])
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(relevant)))])
    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        'precision@k': precision,
        'recall@k': recall,
        'ndcg@k': ndcg
    }


# Get actual items user interacted with in test set
test_user_items = test_df[test_df['user_id'] == test_user_id]['item_id'].values

if len(test_user_items) > 0:
    metrics = compute_metrics(candidate_items, test_user_items, k=10)
    print("✓ Evaluation Metrics (User 0, K=10):")
    for metric, value in metrics.items():
        print(f"  - {metric}: {value:.4f}")
else:
    print("⚠️  User 0 has no test interactions")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)
print("""
✓ Data Generation:        100,000 interactions generated
✓ Feature Engineering:    User and item features created
✓ Train/Test Split:       Time-based split (90/10)
✓ Model Training:         Matrix Factorization trained
✓ Embedding Generation:   64-dimensional embeddings
✓ Candidate Generation:   Top-50 items retrieved via similarity
✓ Evaluation:             Precision, Recall, NDCG computed

Next Steps:
1. Explore src/ directory for production-grade implementations
2. Review INTERVIEW_GUIDE.md for staff-level preparation
3. Study trade-offs in each component
4. Practice explaining design decisions

Key Files:
- src/data_pipeline/data_loader.py         → Data processing at scale
- src/feature_engineering/feature_pipeline.py → Feature engineering
- src/embeddings/embedding_models.py       → Embedding strategies
- src/models/ranking_model.py              → Ranking models
- src/serving/recommendation_service.py    → Production serving
- src/monitoring/monitoring.py             → Observability
- INTERVIEW_GUIDE.md                       → Interview preparation
""")
print("=" * 80)
