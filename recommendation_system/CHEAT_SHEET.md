# 📝 Staff Interview Cheat Sheet - Recommendation Systems

## 🎯 One-Minute Elevator Pitch

"I designed a production recommendation system serving 1B users with sub-100ms latency. It uses a two-stage architecture: first, FAISS-based ANN search retrieves 500 candidates from millions in 20ms, then LightGBM re-ranks them in 15ms. The system processes TBs of data daily with Spark, maintains online/offline feature consistency with a feature store, and includes comprehensive monitoring for data drift and A/B testing. We improved CTR by 75% while maintaining 99th percentile latency under 100ms."

---

## 💡 Top 10 Interview Questions (Speed Round)

### 1. **"Design a recommendation system for [YouTube/Netflix/Amazon]"**

**30-Second Answer:**
"Two-stage retrieval: (1) Candidate generation using two-tower neural network + FAISS for ANN search - reduces millions to hundreds in 20ms, (2) LightGBM ranking with full features - scores hundreds in 15ms. Data pipeline: Kafka for streaming, Spark for batch, feature store for consistency. Monitoring: track CTR/conversion and data drift."

**Draw:** Two-stage architecture diagram

---

### 2. **"How do you handle cold start?"**

**30-Second Answer:**
"Three types: (1) New users - use demographics + popular items + content-based if preferences given, (2) New items - content features + show to sample users for exploration, (3) New system - transfer learning + pre-trained embeddings. Key: hybrid approach combining collaborative + content-based."

---

### 3. **"LightGBM vs Neural Network for ranking?"**

**30-Second Answer:**
"LightGBM for ranking at Google/Meta/Uber because: 10x faster training, better with tabular features, interpretable, robust. Neural networks for candidate generation where we need semantic similarity. Trade-off: NN can learn complex interactions but slower and needs more data. For most production systems, LightGBM wins for ranking."

---

### 4. **"How do you ensure diversity?"**

**30-Second Answer:**
"Post-ranking with sliding window: max N items per category. Alternatives: MMR (maximize marginal relevance), DPP (determinantal point processes). Trade-off: 10-20% diversity boost improves long-term engagement vs pure relevance. Measure with Gini coefficient."

---

### 5. **"How do you choose embedding dimensionality?"**

**30-Second Answer:**
"Formula: dim ≈ 4 * ⁴√(vocab_size). For 1M items → 128 dims. Trade-off: lower dims (32) = faster but less expressive, higher (512) = better accuracy but slower search. Validate empirically with offline metrics. Sweet spot: 128-256 for most systems."

---

### 6. **"How do you detect data drift?"**

**30-Second Answer:**
"PSI (Population Stability Index) for continuous features. PSI > 0.2 = significant drift → trigger retrain. Chi-square test for categoricals. Monitor daily, correlate with performance drops. Also track online metrics (CTR, conversion) for early warning."

---

### 7. **"How do you scale to billions of users?"**

**30-Second Answer:**
"Data: Spark for batch (TBs), Flink for streaming. Features: Feature store with online (Redis) + offline (S3/Parquet). Serving: FAISS on GPU (10M items < 20ms), cache user embeddings (1hr TTL), horizontal scaling (100s servers). Cost: batch predictions for email, real-time for web."

---

### 8. **"How do you ensure online/offline consistency?"**

**30-Second Answer:**
"Feature store (Feast/Tecton): single source of truth, maintains both online and offline, point-in-time correctness. Alternative: log feature values at inference, use for training. Key: same code for both environments, use containers/UDFs."

---

### 9. **"What metrics do you optimize?"**

**30-Second Answer:**
"Offline: NDCG@10 (ranking quality), AUC, coverage. Online: CTR (engagement), conversion rate (business), time-to-click. North star: GMV for e-commerce, watch time for streaming. Important: offline and online don't always correlate - always A/B test!"

---

### 10. **"How do you handle A/B tests?"**

**30-Second Answer:**
"Calculate sample size: for 3% CTR, detect 5% lift → need 100K users per variant. Run for statistical significance (p < 0.05). Guard against multiple testing with Bonferroni correction. Ship if: statistically significant AND business metric improves AND no guardrail violations."

---

## 🏗️ Architecture Quick Draw

```
┌─────────────────────────────────────────────────────┐
│  User Request (user_id, context)                    │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │  FEATURE STORE            │
         │  • User features (Redis)  │
         │  • Item features (cache)  │
         │  • Context features       │
         └─────────────┬─────────────┘
                       │
         ┌─────────────▼─────────────────┐
         │  CANDIDATE GENERATION (20ms)  │
         │  • Two-tower model            │
         │  • User embedding             │
         │  • FAISS ANN search           │
         │  → 500 candidates             │
         └─────────────┬─────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │  RANKING (15ms)           │
         │  • LightGBM               │
         │  • Full features          │
         │  → Re-ranked list         │
         └─────────────┬─────────────┘
                       │
         ┌─────────────▼────────────────┐
         │  BUSINESS LOGIC (5ms)        │
         │  • Diversity                 │
         │  • Freshness                 │
         │  • Deduplication             │
         └─────────────┬────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │  Response (Top-N items)   │
         │  Total: < 50ms            │
         └───────────────────────────┘
```

---

## 📊 Key Numbers to Remember

| Metric | Value | Context |
|--------|-------|---------|
| **Latency (p99)** | < 100ms | Industry standard |
| **Candidate retrieval** | 20-40ms | ANN search |
| **Ranking** | 10-30ms | LightGBM on 500 items |
| **Embedding dim** | 128-256 | Sweet spot |
| **FAISS throughput** | 10M items | Sub-linear search |
| **Feature store latency** | < 10ms | Critical path |
| **Retrain frequency** | Daily-Weekly | Depends on drift |
| **Data retention** | 90-180 days | Training window |
| **PSI threshold** | > 0.2 | Significant drift |
| **A/B test duration** | 1-2 weeks | Statistical power |

---

## 🎓 Technical Depth - 3 Levels

### Level 1: High-Level (Product Manager)
"Two-stage system: fast filtering then precise ranking. Improves CTR by showing more relevant items."

### Level 2: Engineering Manager
"Two-tower network generates embeddings, FAISS enables sub-linear search, LightGBM handles ranking with 100+ features. Sub-100ms p99 latency at scale."

### Level 3: Staff Engineer (You)
"Query tower encodes user dynamically, item tower pre-computed and cached. L2-normalized embeddings enable cosine similarity via dot product. FAISS IVF index with 100 clusters, nprobe=10 for 95% recall. LightGBM lambda-rank objective optimizes NDCG directly. Feature store ensures point-in-time correctness via temporal joins in Spark."

---

## 💭 Thought Process Framework

When asked ANY system design question:

1. **Clarify** (2 min)
   - Scale? (users, items, QPS)
   - Latency requirements?
   - Data characteristics?
   - Constraints?

2. **High-Level** (3 min)
   - Draw two-stage architecture
   - Explain data flow
   - Mention key components

3. **Deep Dive** (20 min)
   - Pick 2-3 components
   - Explain trade-offs
   - Discuss alternatives
   - Show expertise

4. **Scale** (5 min)
   - How to handle 10x scale?
   - Bottlenecks?
   - Cost optimization?

5. **Production** (5 min)
   - Monitoring
   - Failure modes
   - A/B testing

---

## 🚨 Common Pitfalls to Avoid

❌ **DON'T SAY:**
- "Just use a neural network"
- "We'll use K-means clustering"
- "Accuracy is the most important metric"
- "We can compute all user-item pairs"
- "Random train/test split is fine"

✅ **DO SAY:**
- "Trade-off between accuracy and latency..."
- "LightGBM for ranking because..."
- "NDCG is the industry standard because..."
- "Two-stage retrieval for scalability..."
- "Time-based split to prevent leakage..."

---

## 🎯 Buzzwords to Use (Naturally)

### Data Pipeline
- "Point-in-time correctness"
- "Feature store"
- "Online/offline consistency"
- "Streaming + batch lambda architecture"

### Models
- "Two-tower architecture"
- "ANN search"
- "LambdaRank objective"
- "Embedding normalization"
- "Hard negative mining"

### Serving
- "Sub-linear search"
- "FAISS IVF index"
- "Candidate generation"
- "Multi-stage retrieval"
- "Cache-aside pattern"

### Monitoring
- "Data drift"
- "PSI (Population Stability Index)"
- "Online/offline metric correlation"
- "A/B testing framework"
- "Guardrail metrics"

---

## 📚 If You Have 1 Hour Before Interview

**Priority Order:**

1. **15 min:** Read PROJECT_OVERVIEW.md
2. **20 min:** Practice drawing architecture + explaining
3. **15 min:** Review this cheat sheet
4. **10 min:** Memorize key numbers and trade-offs

**Focus On:**
- Two-stage retrieval (why?)
- LightGBM vs NN (when?)
- Cold start (how?)
- Scaling (billions?)

---

## 🎤 Opening Statement Template

"In my experience building recommendation systems at scale, I've found that the key challenge is balancing accuracy with latency. I designed a system using two-stage retrieval - first, a two-tower neural network with FAISS for fast candidate generation, then LightGBM for precise ranking. This architecture allows us to search through millions of items in under 100ms while maintaining high relevance. I can dive deep into any component - the embedding strategy, feature engineering, serving infrastructure, or monitoring approach."

---

## 🏆 Closing Strong

**When Asked: "Any questions for us?"**

✅ **GOOD QUESTIONS:**
- "What's the current latency budget for recommendations?"
- "How do you handle the explore/exploit trade-off?"
- "What's your approach to online learning vs batch retraining?"
- "How do you measure long-term user satisfaction vs short-term engagement?"
- "What's the biggest challenge in your recommendation system right now?"

❌ **AVOID:**
- "What technologies do you use?" (too junior)
- "Do you use machine learning?" (too broad)
- Generic questions about culture/benefits

---

## 🎯 Final Confidence Boosters

**You Know:**
✅ 5 embedding strategies (implemented)
✅ 3 ranking models (LightGBM, DCN, DeepFM)
✅ Production serving architecture
✅ Data drift detection (3 methods)
✅ A/B testing framework
✅ Scalability to billions
✅ Feature engineering for mixed types
✅ Cold start handling (3 scenarios)

**You Can:**
✅ Draw the architecture from memory
✅ Explain trade-offs for each decision
✅ Code the core components
✅ Discuss production challenges
✅ Connect to business metrics

**You Are Ready!** 🚀

---

**Pro Tip:** Print this cheat sheet and review it 10 minutes before interview. Glance at it during "break" if allowed. The numbers and architecture will be fresh in your mind.

Good luck! You've got this! 💪
