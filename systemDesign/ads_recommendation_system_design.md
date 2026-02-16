# ML System Design Interview: Ads Recommendation System
## Senior / Staff MLE — FAANG-Level Interview

---

## Interview Transcript & Solution

---

### 🎙️ Interviewer

> "Design an Ads Recommendation System for a large-scale platform — think Meta, Google, or TikTok. The system should decide which ad to show to a given user in a given context. Walk me through the full end-to-end ML system."

---

### Step 1: Clarifying Questions

**Candidate:** Before I dive in, I'd like to clarify a few things to scope the problem correctly.

**Candidate:**
- **What is the platform surface?** Is this a social media feed (like Facebook/Instagram), a search results page (like Google), or a video platform (like YouTube/TikTok)?
- **What's the scale?** How many users, how many ads in the inventory, and what's the QPS we're targeting?
- **What's the primary business objective?** Are we optimizing for revenue (eCPM), user experience (engagement + relevance), or a blend?
- **Do we have historical data?** Click logs, conversion logs, user profiles, ad metadata?

**Interviewer:** Let's say it's a social media feed — similar to Facebook or Instagram. Assume 2B+ monthly active users, millions of active ad campaigns, ~500K QPS at peak. We want to maximize revenue while maintaining user experience quality. Yes, you have rich historical data.

**Candidate:** Perfect. Let me structure my answer across these key areas:

1. Problem Formulation & Metrics
2. High-Level System Architecture
3. Data & Feature Engineering
4. Model Architecture (Multi-Stage)
5. Training Pipeline
6. Serving Architecture
7. Experimentation & Monitoring

---

### Step 2: Problem Formulation & Metrics

**Candidate:** Let me start by clearly defining what we're optimizing and how we'll measure success.

#### Business Objective

The core objective is to **maximize total ad revenue** while maintaining a healthy user experience. Revenue in an ads system is driven by:

```
Revenue = Σ (bid × P(click) × P(conversion | click))
        = Σ (eCPM)
```

So our ML task is to **accurately predict the probability of user engagement** (click, conversion, etc.) for each (user, ad, context) triple.

#### ML Task Decomposition

I'd decompose this into a multi-task prediction problem:

| Task | Label | Model Output |
|------|-------|-------------|
| Click-Through Rate (CTR) | Did user click? (0/1) | P(click) |
| Conversion Rate (CVR) | Did user convert post-click? (0/1) | P(conversion \| click) |
| Engagement Quality | Did user hide/report ad? (0/1) | P(negative feedback) |
| Long-term Value | Did user return / LTV impact? | Estimated value score |

#### Metrics

**Offline Metrics:**
- AUC-ROC and AUC-PR for CTR and CVR models
- Normalized Cross-Entropy (NCE) — critical for calibration
- Calibration plots (predicted vs. actual rates)
- NDCG for ranking quality

**Online Metrics:**
- Revenue per 1000 impressions (RPM)
- Click-through rate
- Conversion rate
- User negative feedback rate (hide, report)
- Ad load sensitivity (ads per session)
- Long-term user retention (guardrail)

**Candidate:** Calibration is absolutely critical here — unlike pure ranking, ads systems need well-calibrated probabilities because they're multiplied by bids in the auction. A model with great AUC but poor calibration can destroy revenue.

**Interviewer:** Good. That's an important nuance. Walk me through the architecture.

---

### Step 3: High-Level System Architecture

**Candidate:** The system follows a classic **multi-stage funnel architecture** to handle the scale constraint. We can't run a heavyweight model on millions of ad candidates per request.

#### Multi-Stage Funnel

```
┌─────────────────────────────────────────────────────────────────┐
│                     AD REQUEST (User + Context)                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: AD RETRIEVAL / CANDIDATE GENERATION                  │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Targeting   │  │  Embedding   │  │  Inverted Index       │  │
│  │  Filters     │  │  ANN Search  │  │  (Keyword/Interest)   │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
│  Millions of ads  →→→  ~10,000 candidates                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: PRE-RANKING / LIGHTWEIGHT SCORING                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Lightweight Model (Two-Tower / Logistic Regression)     │   │
│  │  Fast inference, coarse-grained features                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ~10,000 candidates  →→→  ~500 candidates                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: RANKING / HEAVY MODEL                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Deep Neural Network (Multi-Task)                        │   │
│  │  Rich cross-features, attention, DCN-v2                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ~500 candidates  →→→  ~50 scored candidates                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: AUCTION & RE-RANKING                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  eCPM = bid × pCTR × pCVR                               │   │
│  │  + Ad quality score                                      │   │
│  │  + Diversity / Pacing / Budget constraints                │   │
│  │  + Negative feedback penalty                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ~50 candidates  →→→  3-5 ads placed in feed                   │
└─────────────────────────────────────────────────────────────────┘
```

**Interviewer:** Why four stages and not just one powerful model?

**Candidate:** Pure latency and compute economics. At 500K QPS with millions of ad candidates per request, the math doesn't work for a single heavy model. Each stage trades off model complexity for throughput:

| Stage | Candidates | Latency Budget | Model Complexity |
|-------|-----------|---------------|-----------------|
| Retrieval | Millions → 10K | <10ms | ANN / simple rules |
| Pre-Ranking | 10K → 500 | <5ms per ad | Lightweight NN |
| Ranking | 500 → 50 | <10ms per ad | Deep multi-task NN |
| Auction | 50 → 3-5 | <2ms | Business logic + LP |

Total end-to-end latency target: **< 100ms** at p99.

---

### Step 4: Data & Feature Engineering

**Candidate:** Features are the lifeblood of an ads system. Let me break them down by category.

#### Feature Taxonomy

```
┌────────────────────────────────────────────────────────────────────┐
│                        FEATURE CATEGORIES                          │
├──────────────────┬──────────────────┬──────────────────────────────┤
│   USER FEATURES  │   AD FEATURES    │   CONTEXT FEATURES           │
├──────────────────┼──────────────────┼──────────────────────────────┤
│ • Demographics   │ • Ad creative    │ • Time of day / day of week  │
│   (age, gender,  │   (image embed., │ • Device type (mobile/web)   │
│    location)     │    text embed.)  │ • Position in feed           │
│ • Interest graph │ • Advertiser ID  │ • Network type (wifi/LTE)    │
│ • Behavioral     │ • Category/Topic │ • Feed density               │
│   sequences      │ • Historical CTR │ • Session depth              │
│ • Purchase hist. │ • Landing page   │ • Preceding content type     │
│ • Social graph   │   quality score  │ • Geo-context (home/work)    │
│   embeddings     │ • Campaign age   │                              │
│ • Engagement     │ • Budget pacing  │                              │
│   patterns       │   ratio          │                              │
└──────────────────┴──────────────────┴──────────────────────────────┘

                    ┌──────────────────────────┐
                    │   CROSS FEATURES          │
                    ├──────────────────────────┤
                    │ • User-Ad affinity score  │
                    │ • User×Category history   │
                    │ • User×Advertiser history │
                    │ • User×Creative-type pref │
                    │ • Social proof features   │
                    │   (did friends engage?)   │
                    └──────────────────────────┘
```

#### Key Feature Engineering Decisions

**Real-time features** (computed at serving time):
- User's last N actions in current session (sequence features)
- Time since last ad impression (ad fatigue signal)
- Current session engagement rate

**Near-real-time features** (updated every few minutes):
- Ad's rolling CTR over last 1h/6h/24h
- User's rolling engagement stats
- Campaign budget utilization ratio

**Batch features** (updated daily):
- User embeddings from social graph
- Long-term interest profiles
- Advertiser quality scores

**Candidate:** A critical design decision is the **feature store architecture**. I'd use a dual-layer feature store:

```
┌──────────────────────────────────────────────────┐
│              FEATURE STORE ARCHITECTURE           │
│                                                   │
│  ┌─────────────────────┐   ┌──────────────────┐  │
│  │   ONLINE STORE      │   │   OFFLINE STORE   │  │
│  │   (Redis / RocksDB) │   │   (Hive / S3)     │  │
│  │                     │   │                    │  │
│  │  • p99 < 5ms reads  │   │  • Training data   │  │
│  │  • User features    │   │  • Feature backfill│  │
│  │  • Real-time stats  │   │  • Point-in-time   │  │
│  │  • Pre-computed     │   │    correctness     │  │
│  │    embeddings       │   │                    │  │
│  └─────────────────────┘   └──────────────────┘  │
│           │                        │              │
│           └──────────┬─────────────┘              │
│                      │                            │
│            Feature consistency                    │
│            (train-serve skew prevention)          │
└──────────────────────────────────────────────────┘
```

**Interviewer:** How do you handle the train-serve skew problem?

**Candidate:** This is one of the biggest practical challenges. Three strategies:

1. **Log-and-replay**: At serving time, we log the exact feature values used for each prediction alongside the impression. Training data is constructed from these logged features — guaranteeing what the model sees in training matches production exactly.

2. **Point-in-time joins**: For batch features, we timestamp everything and do temporal joins so that training examples only use features available at that timestamp.

3. **Feature monitoring**: Continuous distribution monitoring (PSI, KL-divergence) between training and serving feature distributions, with automated alerts.

---

### Step 5: Model Architecture (The Ranking Model — Stage 3)

**Candidate:** The ranking model is where the main ML innovation lives. I'd use a **Multi-Task Deep Neural Network** with several key architectural components.

#### Overall Model Architecture

```
                        ┌──────────────┐  ┌──────────────┐
                        │  P(click)    │  │  P(convert)  │
                        │  (CTR head)  │  │  (CVR head)  │
                        └──────┬───────┘  └──────┬───────┘
                               │                  │
                        ┌──────┴───────┐  ┌──────┴───────┐
                        │  Task-specific│  │Task-specific │
                        │  Tower (MLP) │  │Tower (MLP)   │
                        │  [256→128→64]│  │[256→128→64]  │
                        └──────┬───────┘  └──────┬───────┘
                               │                  │
                               └────────┬─────────┘
                                        │
                               ┌────────┴────────┐
                               │  SHARED BOTTOM   │
                               │  NETWORK         │
                               │                  │
                               │  MMoE Layer      │
                               │  (Multi-gate     │
                               │   Mixture of     │
                               │   Experts)       │
                               └────────┬─────────┘
                                        │
                    ┌───────────────────┬┴┬───────────────────┐
                    │                   │ │                   │
             ┌──────┴──────┐  ┌────────┴─┴──────┐  ┌────────┴────────┐
             │  DCN-v2     │  │  Deep Network    │  │  Sequence Model │
             │  (Cross     │  │  (MLP tower)     │  │  (Transformer / │
             │   Network)  │  │  [1024→512→256]  │  │   DIN / DIEN)   │
             └──────┬──────┘  └────────┬─────────┘  └────────┬────────┘
                    │                  │                      │
                    └──────────────────┼──────────────────────┘
                                       │
                              ┌────────┴─────────┐
                              │  EMBEDDING LAYER  │
                              │                   │
                              │  Sparse features  │
                              │  → Embeddings     │
                              │  (dim: 16-64)     │
                              │                   │
                              │  Dense features   │
                              │  → Normalization  │
                              └────────┬──────────┘
                                       │
                    ┌──────────┬────────┼────────┬──────────┐
                    │          │        │        │          │
                ┌───┴───┐ ┌───┴───┐ ┌──┴──┐ ┌──┴──┐  ┌───┴────┐
                │ User  │ │  Ad   │ │Cross│ │Ctx  │  │Sequence│
                │Features│ │Features│ │Feats│ │Feats│  │Features│
                └───────┘ └───────┘ └─────┘ └─────┘  └────────┘
```

#### Key Architectural Choices & Rationale

**1. Multi-Gate Mixture of Experts (MMoE)**

**Candidate:** I chose MMoE over a simple shared-bottom because CTR and CVR tasks have related but distinct data distributions. MMoE lets each task learn its own gating weights over shared expert sub-networks, giving better task-specific specialization while still sharing useful representations.

```
        Gate_CTR    Gate_CVR
        [w1,w2,w3]  [w1,w2,w3]
            │            │
    ┌───────┼────────────┼───────┐
    │       │            │       │
┌───┴───┐ ┌┴────┐  ┌────┴┐ ┌───┴───┐
│Expert1│ │Exp.2│  │Exp.3│ │Expert4│
│(MLP)  │ │(MLP)│  │(MLP)│ │(MLP)  │
└───┬───┘ └──┬──┘  └──┬──┘ └───┬───┘
    └────────┴────┬────┴────────┘
                  │
           Shared Input
```

**2. DCN-v2 (Deep & Cross Network v2)**

For explicit feature interactions. Unlike raw MLPs which learn interactions implicitly, DCN-v2 explicitly models bounded-degree feature crosses — critical for capturing patterns like "users aged 25-34 in tech industry respond well to SaaS ads on weekday mornings."

**3. Deep Interest Network (DIN) / DIEN for Sequences**

User behavior sequences (last 50 ad interactions) are processed with attention mechanisms where the **candidate ad attends over the user's historical interactions**, giving adaptive user representations that are ad-aware.

```
Attention Score = softmax(Ad_embedding · History_i_embedding)

User_representation = Σ(attention_i × history_embedding_i)
```

This is far superior to average-pooling because it activates different parts of user history depending on which ad is being scored.

**Interviewer:** How do you handle the CVR prediction given that conversions only happen post-click? That's a sample selection bias problem.

**Candidate:** Excellent question. This is a well-known problem — if we only train CVR on clicked samples, the model is biased because the click itself is a confounding filter.

I'd use the **ESMM (Entire Space Multi-Task Model)** approach:

```
                P(click AND convert)
P(convert|click) = ────────────────────
                       P(click)

So we model:
  • pCTR from the CTR tower
  • pCTCVR = P(click AND convert) jointly
  • pCVR = pCTCVR / pCTR

Key insight: pCTCVR is trained on ALL impressions (not just clicks),
which eliminates the sample selection bias.
```

The CVR tower's parameters are trained via the pCTCVR loss computed over the entire impression space, while the final pCVR is derived by division at inference.

---

### Step 6: Training Pipeline

**Candidate:** Let me walk through the training infrastructure.

#### Training Data Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Ad Serving  │────▶│  Impression  │────▶│  Join with        │
│  Logs        │     │  Logger      │     │  Downstream Labels│
└─────────────┘     └──────────────┘     │  (click/convert   │
                                          │   within windows) │
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                          ┌────────────────────┐
                                          │  Feature Snapshot   │
                                          │  (logged features + │
                                          │   point-in-time     │
                                          │   batch features)   │
                                          └────────┬───────────┘
                                                   │
                                                   ▼
                                          ┌────────────────────┐
                                          │  Training Data      │
                                          │  Pipeline           │
                                          │  (negative sampling,│
                                          │   deduplication,    │
                                          │   data validation)  │
                                          └────────┬───────────┘
                                                   │
                              ┌─────────────────────┼──────────────────┐
                              ▼                     ▼                  ▼
                     ┌─────────────┐      ┌─────────────┐    ┌──────────────┐
                     │ Full Retrain│      │ Incremental  │    │ Real-time    │
                     │ (Weekly)    │      │ (Daily)      │    │ (Streaming)  │
                     │ All data    │      │ Last N days  │    │ Online learn │
                     └─────────────┘      └─────────────┘    └──────────────┘
```

#### Training Strategy

**Candidate:** A few critical decisions:

**a) Loss Function:**
```
L_total = α · L_CTR(BCE) + β · L_CTCVR(BCE) + γ · L_neg_feedback(BCE) + λ · L_calibration

Where calibration loss ensures predicted probabilities match observed rates
in bucketed segments.
```

**b) Training Cadence:**
- **Full retrain weekly** on 30 days of data — resets model from scratch
- **Daily incremental warm-start** — fine-tune from the last checkpoint on the latest day's data
- **Optional: Online learning** with streaming updates for ultra-fresh user signals (but adds complexity and instability risk)

**c) Label Attribution Window:**
- Click: attributed within 30 seconds of impression
- Conversion: attributed within 7-day window post-click
- This means training data for CVR is delayed by 7 days (stale label problem)

**Candidate:** The stale label problem is significant. A practical solution is to use a **label correction model** or **importance weighting** where recent but incomplete labels are reweighted based on historical conversion delay curves.

**d) Handling Class Imbalance:**
- CTR is typically ~2-5%, CVR ~1-3% of clicks
- Use **negative downsampling** during training (e.g., keep 10% of negatives)
- Apply **calibration correction** at serving: `p_calibrated = p / (p + (1-p)/w)` where w is the downsampling rate

---

### Step 7: Serving Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    SERVING INFRASTRUCTURE                          │
│                                                                    │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐    │
│  │  Client   │───▶│  Ad Request  │───▶│  Feature Assembly     │    │
│  │  (App)    │    │  Router      │    │  Service               │    │
│  └──────────┘    └──────────────┘    │  ┌──────────────────┐ │    │
│                                       │  │ Online Feature   │ │    │
│                                       │  │ Store (Redis)    │ │    │
│                                       │  └──────────────────┘ │    │
│                                       │  ┌──────────────────┐ │    │
│                                       │  │ Real-time Feature│ │    │
│                                       │  │ Compute          │ │    │
│                                       │  └──────────────────┘ │    │
│                                       └───────────┬───────────┘    │
│                                                   │                │
│                                                   ▼                │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │               MODEL SERVING LAYER                         │    │
│  │  ┌────────────┐  ┌─────────────┐  ┌────────────────┐     │    │
│  │  │ Retrieval  │─▶│ Pre-Ranker  │─▶│ Heavy Ranker   │     │    │
│  │  │ (FAISS/    │  │ (TF-Serving)│  │ (TF-Serving /  │     │    │
│  │  │  ScaNN)    │  │             │  │  Triton)       │     │    │
│  │  └────────────┘  └─────────────┘  └────────────────┘     │    │
│  │                                           │               │    │
│  │              GPU Cluster (batched inference)               │    │
│  └───────────────────────────────────────────┬───────────────┘    │
│                                              │                    │
│                                              ▼                    │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │               AUCTION ENGINE                              │    │
│  │  • eCPM scoring    • Budget pacing   • Frequency caps     │    │
│  │  • Diversity rules  • Policy filters  • Ad quality gates  │    │
│  └───────────────────────────────────────┬───────────────────┘    │
│                                          │                        │
│                                          ▼                        │
│                                   ┌─────────────┐                │
│                                   │  Response:   │                │
│                                   │  Top K Ads   │                │
│                                   └─────────────┘                │
└────────────────────────────────────────────────────────────────────┘
```

#### Serving Optimizations

**Candidate:** At 500K QPS, every millisecond matters. Key optimizations:

1. **Model quantization**: INT8 quantization for the embedding tables (which dominate model size) — reduces memory 4x with <0.1% AUC loss.

2. **Batched GPU inference**: Batch requests on the GPU side to maximize throughput. Dynamic batching with timeout to trade latency for throughput.

3. **Pre-computed user embeddings**: The user tower of the two-tower model is independent of the ad — compute it once per request and reuse across all ad candidates.

4. **Embedding table sharding**: The embedding tables for sparse features (user IDs, ad IDs) can be hundreds of GB. Shard across multiple servers with an embedding lookup service.

5. **Cascading timeout**: If the heavy ranker times out, fall back to pre-ranker scores rather than returning no ads.

---

### Step 8: Auction Mechanism

**Candidate:** The auction is where ML meets business logic.

#### Generalized Second-Price (GSP) → VCG-style Auction

```
For each ad slot position k:

  Score(ad_i) = bid_i × pCTR_i × pCVR_i × quality_i - penalty_i

  Where:
    bid_i        = advertiser's bid (CPC or CPA)
    pCTR_i       = predicted click-through rate
    pCVR_i       = predicted conversion rate
    quality_i    = ad creative quality score (0.8 - 1.2 multiplier)
    penalty_i    = negative feedback prediction × penalty weight

  Winner pays: Score(2nd place) / pCTR_winner  (GSP pricing)
```

**Key business constraints applied post-ranking:**
- **Budget pacing**: Smooth delivery across the day (don't blow budget at 9am)
- **Frequency capping**: Max N impressions per user per ad per day
- **Diversity**: Don't show 3 ads from the same advertiser in sequence
- **Policy compliance**: Filter out policy-violating ads

---

### Step 9: Experimentation & Monitoring

**Candidate:** This is where many systems fail in practice.

#### A/B Testing Framework

```
┌──────────────────────────────────────────────────────────┐
│                 A/B TESTING PIPELINE                      │
│                                                          │
│   User traffic                                           │
│       │                                                  │
│       ▼                                                  │
│   ┌────────────────────┐                                 │
│   │  Hash-based bucket │                                 │
│   │  (user_id mod N)   │                                 │
│   └────────┬───────────┘                                 │
│            │                                             │
│      ┌─────┼─────┐                                       │
│      ▼     ▼     ▼                                       │
│   Control  Trt1  Trt2                                    │
│    (80%)  (10%) (10%)                                    │
│      │     │     │                                       │
│      ▼     ▼     ▼                                       │
│   ┌────────────────────┐                                 │
│   │  Statistical test: │  Primary: Revenue per user      │
│   │  • Revenue lift    │  Guardrail: NPS, retention,     │
│   │  • Confidence      │            negative feedback    │
│   │  • Power analysis  │  Duration: 1-2 weeks            │
│   └────────────────────┘                                 │
└──────────────────────────────────────────────────────────┘
```

#### Real-Time Monitoring Dashboard

**Key monitoring signals:**

| Signal | Frequency | Alert Threshold |
|--------|-----------|----------------|
| Overall CTR | Per-minute | ±10% from baseline |
| Revenue RPM | Per-minute | ±15% from baseline |
| Model latency (p50/p99) | Per-second | p99 > 80ms |
| Feature freshness | Per-minute | Staleness > 10min |
| Prediction distribution shift | Hourly | PSI > 0.1 |
| Negative feedback rate | Per-minute | +20% from baseline |
| Serving errors / timeouts | Per-second | > 0.1% error rate |

**Candidate:** I'd also build an **automated circuit breaker**: if the model's prediction distribution shifts dramatically (possibly due to a bad model push or data pipeline failure), automatically roll back to the previous model version.

---

### Step 10: Advanced Topics & Staff-Level Depth

**Interviewer:** You've covered the system well. Let me push you on a few advanced areas.

#### 10a. Cold Start Problem

**Interviewer:** How do you handle new ads with no engagement history?

**Candidate:** Multi-pronged approach:

1. **Content-based features**: Extract features from the ad creative (image embeddings from a pre-trained vision model, text embeddings from BERT/LLM). These generalize to new ads immediately.

2. **Explore-exploit**: Use a Thompson Sampling or Upper Confidence Bound (UCB) approach in the auction. New ads get an exploration bonus that decays as we collect data:

```
exploration_bonus = α × sqrt(log(total_impressions) / (ad_impressions + 1))
```

3. **Hierarchical priors**: Share statistics at the advertiser → campaign → ad group level. A new ad from a known advertiser starts with the advertiser's average CTR as a prior, then personalizes with Bayesian updating.

4. **Dedicated exploration budget**: Reserve 5-10% of ad inventory for exploration, ensuring new ads get sufficient impressions to learn from.

#### 10b. Position Bias

**Interviewer:** How do you handle the fact that higher positions in the feed get more clicks regardless of ad quality?

**Candidate:** Position bias is a major confound. I'd address it at both training and serving time:

**Training time**: Use position as an input feature but apply **position debiasing**:
```
P(click) = σ(f_relevance(user, ad) + g_position(position))

At serving time: set position = default_position for all candidates,
so ranking is based purely on f_relevance.
```

Alternatively, use **inverse propensity weighting (IPW)** where each training example is weighted by 1/P(position | ad was shown there), estimated from position randomization experiments.

#### 10c. Privacy & Personalization Trade-off

**Candidate:** With the deprecation of third-party cookies and increasing privacy regulations (GDPR, CCPA, ATT on iOS), I'd invest in:

1. **On-device models**: Lightweight models that run on the client device using first-party data, sending only encrypted prediction scores to the server.

2. **Federated Learning**: Train on user data without centralizing it. Practical for learning user preferences without raw data leaving the device.

3. **Contextual targeting**: Invest heavily in context features (page content, time, device) that don't require personal data but still provide signal.

4. **Privacy-preserving ML**: Techniques like differential privacy in training, secure aggregation for federated models.

---

### Summary & Closing

**Candidate:** Let me summarize the key design decisions:

```
┌────────────────────────────────────────────────────────────────┐
│              DESIGN DECISIONS SUMMARY                          │
├────────────────────────┬───────────────────────────────────────┤
│ Problem Formulation    │ Multi-task: CTR + CVR + neg feedback  │
│ Architecture           │ 4-stage funnel (retrieve → rank)      │
│ Core Model             │ MMoE + DCN-v2 + DIN attention         │
│ Training               │ Weekly full + daily incremental       │
│ Key Innovation         │ ESMM for unbiased CVR estimation      │
│ Serving                │ Batched GPU, INT8, pre-computed embeds│
│ Auction                │ GSP with quality & penalty modifiers  │
│ Cold Start             │ Explore-exploit + hierarchical priors │
│ Calibration            │ NCE + post-hoc isotonic regression    │
│ Monitoring             │ Real-time PSI + automated rollback    │
└────────────────────────┴───────────────────────────────────────┘
```

**Interviewer:** Great answer. You've demonstrated strong breadth across the full system and depth in critical areas like calibration, CVR bias, and position debiasing. Thank you.

---

*This document represents a Staff/Senior MLE-level system design response covering the full spectrum from problem formulation to production deployment, suitable for FAANG ads ML interviews.*
