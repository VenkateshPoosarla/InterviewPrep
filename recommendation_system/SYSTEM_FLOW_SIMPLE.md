# Recommendation System - Complete Flow

## End-to-End Request Flow (57ms Total)

```
USER REQUEST
    │
    ├─> User opens app, requests recommendations
    │   Input: user_id, device, location, num_items
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 1: GET USER EMBEDDING (5ms)                        │
├─────────────────────────────────────────────────────────┤
│ • Check Redis cache for user embedding                   │
│ • If found: Return cached 128-dim vector (1ms)           │
│ • If not: Fetch features → Encode → Cache (10ms)         │
│ Output: [0.23, -0.45, 0.89, ..., 0.12]                  │
└─────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: CANDIDATE GENERATION (20ms)                      │
├─────────────────────────────────────────────────────────┤
│ • Normalize user embedding (L2 norm)                     │
│ • FAISS ANN search on 10M item embeddings                │
│ • Returns top 500 most similar items                     │
│ • Apply filters (in-stock, region, recently shown)       │
│ Output: 500 candidate items                              │
└─────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: FETCH FEATURES (10ms - PARALLEL)                 │
├─────────────────────────────────────────────────────────┤
│ Thread 1: User features (age, CTR, preferences) - 8ms    │
│ Thread 2: Item features (metadata, popularity) - 6ms     │
│ Thread 3: Context features (time, device) - 2ms          │
│ Output: Feature matrix [500 items × 120 features]        │
└─────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: RANKING (15ms)                                   │
├─────────────────────────────────────────────────────────┤
│ • Create feature vectors for each candidate              │
│ • LightGBM batch prediction (500 items)                  │
│ • Predict CTR/engagement score for each item             │
│ • Sort by predicted score                                │
│ Output: Ranked 500 items with scores                     │
└─────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 5: BUSINESS LOGIC (5ms)                             │
├─────────────────────────────────────────────────────────┤
│ • Diversity: Max 3 items per category                    │
│ • Freshness: Boost new items (age-based)                 │
│ • Deduplication: Remove recently shown items             │
│ • Safety: Apply content filters                          │
│ Output: Final 20 items                                   │
└─────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 6: FORMAT RESPONSE (2ms)                            │
├─────────────────────────────────────────────────────────┤
│ • Fetch display metadata (title, thumbnail, price)       │
│ • Format JSON response                                   │
│ • Log request (async)                                    │
│ Output: JSON with 20 ranked recommendations              │
└─────────────────────────────────────────────────────────┘
    ▼
RESPONSE SENT TO USER
```

## Data Flow (Offline - Daily)

```
DAY N-1 DATA
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 00:00-02:00: DATA COLLECTION                             │
├─────────────────────────────────────────────────────────┤
│ • Collect last 7 days interactions (S3/Parquet)          │
│ • Join user profiles, item metadata                      │
│ • Data validation (remove nulls, duplicates)             │
│ • Time-based split (train/val/test)                      │
│ Output: 10B clean interaction records                    │
└─────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────┐
│ 02:00-04:00: FEATURE ENGINEERING                         │
├─────────────────────────────────────────────────────────┤
│ • Compute user stats (CTR, recency, preferences)         │
│ • Compute item stats (popularity, trending)              │
│ • Create sequences (last 50 interactions per user)       │
│ • Extract text embeddings (BERT)                         │
│ Output: Feature-rich dataset (100+ features)             │
└─────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────┐
│ 04:00-10:00: MODEL TRAINING                              │
├─────────────────────────────────────────────────────────┤
│ Model 1: Two-Tower Embeddings (4 hours, 4 GPUs)         │
│   • User tower: User features → 128-dim embedding        │
│   • Item tower: Item features → 128-dim embedding        │
│   • Output: User & item embeddings                       │
│                                                          │
│ Model 2: LightGBM Ranker (2 hours, 16 CPUs)             │
│   • Input: 100+ features                                 │
│   • Objective: Predict CTR                               │
│   • Output: Ranking model                                │
└─────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────┐
│ 10:00-11:00: EVALUATION                                  │
├─────────────────────────────────────────────────────────┤
│ • Test on holdout set                                    │
│ • Metrics: AUC, NDCG@10, Log Loss                        │
│ • Compare with baseline                                  │
│ • Approve if metrics improved                            │
└─────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────┐
│ 11:00-12:00: DEPLOYMENT                                  │
├─────────────────────────────────────────────────────────┤
│ • Upload models to S3                                    │
│ • Build FAISS index with item embeddings                 │
│ • Deploy to staging → Test                               │
│ • A/B test on 5% traffic                                 │
│ • Gradual rollout if successful (5%→25%→50%→100%)        │
└─────────────────────────────────────────────────────────┘
    ▼
MODELS IN PRODUCTION
```

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER REQUEST                          │
│                   (10K QPS)                              │
└────────────────────────┬────────────────────────────────┘
                         ▼
                  Load Balancer
                         ▼
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   API Server 1  ...  API Server N (100 replicas)
        │                                 │
        └────────────────┬────────────────┘
                         ▼
        ┌────────────────┴────────────────┐
        ▼                ▼                ▼
    Redis Cache    FAISS Index    Feature Store
    (User embs)    (Item embs)      (Feast)
        │                │                │
        └────────────────┴────────────────┘
                         ▼
              LightGBM Model Serving
                         ▼
                  RESPONSE (20 items)
```

## Two-Stage Architecture

```
Stage 1: CANDIDATE GENERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  10,000,000 items
Method: Embedding similarity (FAISS ANN search)
Time:   20ms
Output: 500 candidates

Stage 2: RANKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  500 candidates
Method: LightGBM with 100+ features
Time:   15ms
Output: Ranked 50 items

Stage 3: BUSINESS LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  50 ranked items
Method: Diversity, freshness, dedup, safety filters
Time:   5ms
Output: Final 20 recommendations
```

## Key Components

**Data Pipeline:**
- Sources: User interactions (S3), User profiles (DB), Item metadata (DB)
- Processing: PySpark for ETL, validation, feature engineering
- Output: Clean training data with 100+ features

**Feature Store (Feast):**
- Online: Redis (< 5ms lookup)
- Offline: S3/Parquet (training data)
- Ensures train/serve consistency

**Embedding Models (Two-Tower):**
- User Tower: User features → 128-dim vector
- Item Tower: Item features → 128-dim vector
- Similarity: Cosine (dot product of normalized vectors)

**ANN Search (FAISS):**
- Index type: IVF (Inverted File)
- Items: 10M item embeddings
- Search time: 20ms for top 500
- Recall: 95%+

**Ranking Model (LightGBM):**
- Features: 100+ (user, item, context, cross)
- Objective: Predict CTR
- Inference: 15ms for 500 items
- Metrics: AUC 0.79, NDCG@10 0.82

**Serving:**
- API: FastAPI (async)
- Replicas: 100+ (auto-scaling)
- Cache: Redis (99% hit rate)
- Latency p99: < 100ms (actual: 57ms)

**Monitoring:**
- Metrics: CTR, revenue, latency, error rate
- Drift detection: PSI (Population Stability Index)
- Alerting: Prometheus + Grafana
- A/B testing: 5% → 100% gradual rollout

## Latency Breakdown

```
User Embedding:      5ms  (9%)
Candidate Gen:      20ms  (35%)
Feature Fetch:      10ms  (18%)
Ranking:            15ms  (26%)
Business Logic:      5ms  (9%)
Response Format:     2ms  (3%)
─────────────────────────────
TOTAL:              57ms  (100%)
```

## Daily Metrics

**Volume:**
- Requests: 10 billion/day
- Users: 100 million DAU
- Items: 10 million in catalog
- Training data: 100TB

**Performance:**
- Latency p99: 48ms
- Throughput: 12K QPS per replica
- Cache hit rate: 99%
- Uptime: 99.95%

**Business:**
- CTR: 4.2%
- Conversion: 1.8%
- Revenue: $126K/day
- eCPM: $12.60

**Model:**
- AUC: 0.79
- NDCG@10: 0.82
- Retraining: Daily
- Deployment: A/B tested
