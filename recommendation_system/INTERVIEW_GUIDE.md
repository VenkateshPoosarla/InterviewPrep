# Staff-Level ML Interview Preparation Guide
## Recommendation Systems Deep Dive

---

## Table of Contents
1. [System Design Questions](#system-design)
2. [Technical Deep Dives](#technical-deep-dives)
3. [Trade-offs and Decisions](#trade-offs)
4. [Production ML](#production-ml)
5. [Common Interview Questions](#common-questions)

---

## System Design Questions

### Q1: Design a recommendation system for YouTube/Netflix/E-commerce

**Expected Answer Structure:**

1. **Requirements Clarification**
   - Scale: 1B users, 10M items
   - Latency: < 100ms p99
   - Freshness: Real-time personalization
   - Cold start: Handle new users/items

2. **High-Level Architecture**
   ```
   User Request → Candidate Generation (ANN) → Ranking (ML Model) → Business Logic → Response
   ```

3. **Two-Stage Retrieval** (Critical!)
   - **Why?** Can't run complex model on millions of items in real-time
   - Stage 1: Fast retrieval (ANN on embeddings) - reduces millions to hundreds
   - Stage 2: Precise ranking (complex model) - scores hundreds

4. **Data Pipeline**
   - Batch: Historical data processing (Spark/Beam)
   - Stream: Real-time events (Kafka/Flink)
   - Feature Store: Online/offline consistency (Feast/Tecton)

5. **Model Architecture**
   - Candidate Generation: Two-tower neural network
   - Ranking: LightGBM or DeepFM
   - Embeddings: 128-256 dimensions

6. **Serving**
   - FAISS/ScaNN for vector similarity search
   - Redis for feature caching
   - Model serving: TensorFlow Serving/TorchServe

7. **Monitoring**
   - Online metrics: CTR, conversion, engagement
   - Offline metrics: NDCG, AUC
   - Data drift detection: PSI, KL divergence
   - A/B testing framework

**Follow-up Questions to Expect:**
- How do you handle cold start?
- What if a user has 10,000 items in their history?
- How do you ensure diversity?
- How do you retrain the model?

---

## Technical Deep Dives

### Embedding Generation

**Q: Why embeddings? What are the alternatives?**

**Answer:**
- **Embeddings**: Dense vector representations (64-256 dims)
  - Pros: Capture semantic similarity, enable fast ANN search, generalize well
  - Cons: Less interpretable, require training

- **Alternatives:**
  - Collaborative Filtering (user-item matrix): O(users × items) memory, sparse
  - Content-based (TF-IDF): Interpretable, but doesn't capture interactions
  - Graph-based: Expensive computation

**Q: How do you choose embedding dimensionality?**

**Answer:**
Trade-off analysis:
- **Low dims (32-64)**: Fast, low memory, less expressive
- **High dims (256-512)**: More expressive, slower, more memory
- **Sweet spot (128)**: Balance for most systems

Rule of thumb: `dim = 4 * ⁴√(vocab_size)`

Example:
- 1M items → dim ≈ 126
- 10M items → dim ≈ 178

Validate empirically with offline metrics!

---

### Two-Tower vs Single Tower

**Q: Why use two-tower architecture?**

**Answer:**

**Two-Tower Advantages:**
1. **Independent encoding**: User and item encoded separately
2. **Cacheable item embeddings**: Compute once, reuse for all users
3. **Fast serving**: Pre-compute item embeddings, only encode user at inference
4. **Scalable**: Can update user tower without reprocessing items

**Single Tower Disadvantages:**
1. Must encode every user-item pair → O(users × items) at inference
2. Cannot cache intermediate results
3. Slower for large catalogs

**When to use Single Tower:**
- Small catalog (< 10K items)
- Need complex user-item interactions
- Ranking stage (not retrieval)

---

### Feature Engineering

**Q: How do you handle high-cardinality categorical features?**

**Answer:**

**Strategies:**
1. **Embedding layers** (Best for deep learning)
   - Learn dense representations
   - Dimension: `min(50, vocab_size // 2)`

2. **Frequency-based bucketing**
   - Group rare values into "OTHER"
   - Threshold: min_count = 10-100

3. **Hashing trick**
   - Hash to fixed number of buckets
   - Pros: Fixed memory, handles unseen values
   - Cons: Collisions

4. **Target encoding**
   - Replace category with mean target value
   - Risk: Data leakage if not done carefully

**Example:**
```python
# Brand feature: 100K unique brands
# Most brands appear < 10 times

# Option 1: Keep top 1000, rest → "OTHER"
# Option 2: Embedding layer with dim=64
# Option 3: Hash to 10K buckets
```

**Follow-up: How to prevent data leakage in target encoding?**
- Use out-of-fold encoding (cross-validation)
- Add smoothing (Bayesian approach)
- Only use training data to compute encodings

---

## Trade-offs and Decisions

### Q: Matrix Factorization vs Deep Learning - When to use what?

**Answer:**

| Aspect | Matrix Factorization | Deep Learning |
|--------|---------------------|---------------|
| **Data Size** | < 100M interactions | > 100M interactions |
| **Side Information** | No | Yes (metadata, context) |
| **Training Speed** | Fast (minutes) | Slow (hours/days) |
| **Interpretability** | High | Low |
| **Cold Start** | Poor | Better (can use features) |
| **Complexity** | Low | High |

**Use Matrix Factorization when:**
- Simple collaborative filtering
- Limited data
- Need interpretability
- Good baseline

**Use Deep Learning when:**
- Rich features (text, images, context)
- Large-scale data
- Need complex patterns
- Can afford training cost

**Pro Tip:** Start with MF as baseline, then move to DL if needed!

---

### Q: LightGBM vs Neural Network for Ranking?

**Answer:**

**LightGBM Advantages:**
1. **Faster training** (10x-100x faster)
2. **Better with tabular data** (mixed types)
3. **Interpretable** (feature importance)
4. **Robust** (no need for normalization, handles missing)
5. **Low latency** (< 5ms for 500 candidates)

**Neural Network Advantages:**
1. **Better with high-dim data** (images, text)
2. **Learns interactions** (automatic feature crossing)
3. **End-to-end learning** (with embeddings)

**Industry Practice:**
- **Most companies use LightGBM/XGBoost for ranking** (Google, Meta, Uber)
- Neural networks for candidate generation (two-tower)

**Hybrid Approach:**
- Use NN to generate embeddings
- Feed embeddings as features to LightGBM

---

## Production ML

### Q: How do you ensure online/offline consistency?

**Answer:**

**The Problem:**
- Model trained on offline features (batch computed)
- Serving uses online features (real-time computed)
- Inconsistency → performance degradation

**Solutions:**

1. **Feature Store** (Best practice)
   - Single source of truth for features
   - Maintains online and offline versions
   - Point-in-time correctness for training
   - Examples: Feast, Tecton, AWS SageMaker Feature Store

2. **Logging Feature Values**
   - Log exact feature values used at inference
   - Use logged features for training
   - Trade-off: Storage cost

3. **Shared Feature Logic**
   - Same code for online and offline
   - Package features as library
   - Use containers/UDFs in Spark

**Code Example:**
```python
# feature_store.py (shared)
def compute_user_features(user_data):
    return {
        'age_group': user_data['age'] // 10,
        'total_purchases': user_data['purchases'],
        # ...
    }

# Training (offline)
features = spark_df.mapPartitions(compute_user_features)

# Serving (online)
features = compute_user_features(user_record)
```

---

### Q: How do you handle model updates without downtime?

**Answer:**

**Blue-Green Deployment:**
1. Deploy new model (green) alongside old (blue)
2. Route small % of traffic to green (shadow mode)
3. Compare metrics
4. Gradually shift traffic if successful
5. Keep blue as rollback option

**Canary Deployment:**
1. Deploy to 1% of users
2. Monitor closely for 24 hours
3. Increase to 10%, 50%, 100% if healthy

**A/B Testing:**
1. New model as treatment variant
2. Statistical comparison
3. Ship if significant improvement

**Model Versioning:**
```python
models = {
    'v1.0': load_model('model_v1.pkl'),
    'v1.1': load_model('model_v1.1.pkl'),
}

# Route based on experiment assignment
model_version = get_experiment_variant(user_id)
model = models[model_version]
predictions = model.predict(features)
```

---

### Q: How do you detect and handle data drift?

**Answer:**

**Detection Methods:**

1. **Population Stability Index (PSI)**
   ```python
   psi = sum((current% - baseline%) * ln(current% / baseline%))

   # Thresholds:
   # < 0.1: No change
   # 0.1-0.2: Moderate drift
   # > 0.2: Significant drift → Retrain!
   ```

2. **KL Divergence** (for continuous features)
3. **Chi-square test** (for categorical features)

**Mitigation:**

1. **Automated Retraining**
   ```python
   if psi > 0.2:
       trigger_retraining_pipeline()
   ```

2. **Online Learning**
   - Incrementally update model with new data
   - Use for simple models (linear, logistic)
   - Risk: Drift in learned patterns

3. **Ensemble with Fresh Model**
   - Keep old model (weight: 0.7)
   - Add new model (weight: 0.3)
   - Gradually shift weights

**Monitoring Dashboard:**
- Track PSI for key features daily
- Alert if > threshold
- Correlate with performance metrics

---

## Common Questions

### Q: How do you handle the cold start problem?

**Answer:**

**Three Types:**

1. **New User** (no interaction history)
   - Use demographic features (age, location)
   - Popular items in user's country/segment
   - Content-based (if user provides preferences)
   - Prompt for explicit preferences

2. **New Item** (no interactions yet)
   - Content-based features (category, brand, text)
   - Similar items (based on metadata)
   - Explore/Exploit: Show to sample of users
   - Boost in recommendations (freshness bonus)

3. **New System** (sparse data overall)
   - Transfer learning from similar domain
   - Pre-trained embeddings (text, images)
   - Hybrid: Content + collaborative

**Code Example:**
```python
def get_user_embedding(user_id):
    if user_history[user_id]:
        return learned_embedding[user_id]
    else:
        # Cold start: use demographic features
        demo_features = get_demographics(user_id)
        return demographic_model.encode(demo_features)
```

---

### Q: How do you ensure diversity in recommendations?

**Answer:**

**Why Diversity Matters:**
- Filter bubble problem
- User satisfaction (avoid repetition)
- Business goals (expose long-tail items)

**Techniques:**

1. **MMR (Maximal Marginal Relevance)**
   ```
   Score_final = λ * Relevance - (1-λ) * Similarity_to_selected
   ```
   Iteratively select items that are relevant but dissimilar to already selected

2. **Sliding Window**
   ```python
   category_counts = {}
   for item in ranked_items:
       if category_counts[item.category] < 3:
           results.append(item)
           category_counts[item.category] += 1
   ```

3. **DPP (Determinantal Point Processes)**
   - Probabilistic model for diverse subsets
   - Balances quality and diversity
   - Computationally expensive

4. **Post-Processing Rules**
   - Max N items per category
   - Interleave different types
   - Boost underrepresented categories

**Trade-off:** Diversity vs Relevance
- Too diverse → lower engagement
- Too similar → filter bubble
- **Sweet spot:** 10-20% diversity boost in position

---

### Q: What metrics do you optimize for?

**Answer:**

**Offline Metrics** (Model development):
- **NDCG@K**: Measures ranking quality with position discount
- **MAP@K**: Mean Average Precision
- **AUC-ROC**: Binary classification quality
- **Coverage**: % of catalog recommended
- **Diversity**: Gini coefficient, entropy

**Online Metrics** (Production):
- **CTR**: Click-through rate (engagement)
- **Conversion Rate**: Purchases / impressions
- **Time to Click**: How quickly users engage
- **Session Duration**: Total engagement time
- **Revenue**: Business impact ($$$)

**North Star Metric:**
- E-commerce: GMV (Gross Merchandise Value)
- Streaming: Watch time
- Social: Daily Active Users (DAU)

**Important:** Offline and online metrics don't always correlate!
- Optimize offline metrics during development
- A/B test and validate with online metrics
- Make decisions based on business metrics

---

### Q: How do you scale to billions of users/items?

**Answer:**

**Data Pipeline:**
1. **Distributed Processing**
   - Spark for batch (TBs of data)
   - Flink for streaming (millions events/sec)
   - Partitioning by user_id

2. **Feature Store**
   - Offline: S3/GCS (parquet, partitioned)
   - Online: Redis/DynamoDB (< 10ms latency)

**Model Training:**
1. **Distributed Training**
   - Data parallelism (split data across GPUs)
   - Model parallelism (for huge models)
   - Tools: Horovod, PyTorch DDP

2. **Sampling Strategies**
   - Don't need all negatives
   - Hard negative mining
   - In-batch negatives

**Serving:**
1. **ANN Search**
   - FAISS on GPU: 10M items in < 20ms
   - Sharding: Distribute index across machines
   - Approximation: Trade accuracy for speed

2. **Caching**
   - User embeddings: Redis (1 hour TTL)
   - Item embeddings: Pre-computed, in-memory
   - Popular items: CDN

3. **Load Balancing**
   - Horizontal scaling (100s of servers)
   - Request routing (consistent hashing)

**Cost Optimization:**
- Batch predictions for email campaigns
- Real-time for web/mobile
- Tiered serving (simple model for 90%, complex for 10%)

---

## Behavioral/Leadership Questions (Staff Level)

### Q: Tell me about a time you made a significant technical decision

**STAR Format Example:**

**Situation:** Recommendation CTR was 2%, competitor had 4%. Needed to improve.

**Task:** My task was to redesign the recommendation system to double CTR.

**Action:**
1. **Investigation**: Analyzed offline vs online metrics - poor correlation
2. **Hypothesis**: Model optimizing wrong objective (impressions vs clicks)
3. **Proposal**:
   - Switched from implicit feedback to explicit clicks as labels
   - Implemented two-tower architecture for better candidate generation
   - Added business logic layer for diversity
4. **Validation**: A/B tested with 5% of traffic
5. **Rollout**: Gradual rollout over 2 weeks

**Result:**
- CTR improved from 2% → 3.5% (+75%)
- Engagement time +20%
- Shipped to 100% after 4 weeks

**Learning:** Always validate offline improvements with online experiments!

---

### Q: How do you handle disagreement with stakeholders?

**Example:**

**Situation:** PM wanted to add more recommendations (50 items instead of 20) to increase engagement.

**Disagreement:** I believed this would hurt user experience and long-term retention.

**Approach:**
1. **Listen**: Understood PM's goal (increase engagement metrics)
2. **Data**: Analyzed historical data - more items → lower CTR
3. **Hypothesis**: Users have decision fatigue with too many options
4. **Proposal**: A/B test different counts (20, 35, 50)
5. **Compromise**: Test PM's hypothesis scientifically

**Outcome:**
- Test showed 35 items was optimal (CTR -5%, but session time +10%)
- Shipped 35 items
- Both of us learned from data

**Key Principle:** Disagree and commit, but validate with data!

---

## Red Flags to Avoid

1. **Don't** say "just use a neural network" without justification
2. **Don't** ignore latency constraints (P99 < 100ms is critical)
3. **Don't** forget about cold start problem
4. **Don't** optimize only for accuracy (business metrics matter!)
5. **Don't** neglect monitoring and observability
6. **Don't** ignore bias and fairness issues
7. **Don't** over-engineer (start simple, iterate)

---

## Resources for Further Study

1. **Papers:**
   - Two-Tower: "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (Google, 2019)
   - DCN: "Deep & Cross Network for Ad Click Predictions" (Google, 2017)
   - BERT4Rec: "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"

2. **Blogs:**
   - Netflix Tech Blog: Recommendation Systems
   - Uber Engineering: Personalization at Scale
   - Meta Engineering: Graph-based Recommendations

3. **Courses:**
   - Stanford CS246: Mining Massive Datasets
   - Google's ML Crash Course

4. **Books:**
   - "Recommender Systems Handbook" (Ricci et al.)
   - "Deep Learning for Recommender Systems" (Mu Li)

---

## Final Tips for Staff-Level Interviews

1. **Think End-to-End**: Not just model, but entire system (data → model → serving → monitoring)
2. **Quantify Everything**: Use numbers (latency, scale, metrics)
3. **Discuss Trade-offs**: No perfect solution, explain pros/cons
4. **Show Business Impact**: Connect technical decisions to business outcomes
5. **Demonstrate Leadership**: How you influenced others, drove decisions
6. **Be Honest**: Say "I don't know, but here's how I'd figure it out"
7. **Ask Questions**: Clarify requirements before diving in

Good luck! 🚀
