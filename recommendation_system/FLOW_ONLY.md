# Recommendation System Flow

## 1. User Makes Request

```
User opens app → API Request
Input: user_id, num_items=20
```

## 2. Get User Embedding (5ms)

```
Check Redis cache → Get user's 128-number vector
If not cached: Compute from user's features → Store in cache
```

## 3. Find Similar Items (20ms)

```
Use FAISS to search 10 million items
Find 500 most similar to user
Filter out unavailable/recently shown
```

## 4. Get Features (10ms)

```
Parallel fetch:
- User data (age, preferences, history)
- Item data (price, popularity, ratings)
- Context (time of day, device type)
```

## 5. Rank Items (15ms)

```
LightGBM model scores all 500 items
Predicts click probability for each
Sort by score (highest first)
```

## 6. Apply Business Rules (5ms)

```
Ensure diversity (max 3 per category)
Boost fresh content
Remove duplicates
Apply safety filters
```

## 7. Return Results (2ms)

```
Take top 20 items
Format as JSON
Send to user
```

**Total Time: 57ms**

---

## Data Flow (Happens Daily at Night)

### Collect Data (2 hours)
```
Get last 7 days of user interactions
Clean and validate data
Split into train/validation/test
```

### Create Features (2 hours)
```
Calculate user statistics
Calculate item statistics
Build user behavior sequences
```

### Train Models (6 hours)
```
Train embedding model (creates user/item vectors)
Train ranking model (predicts clicks)
```

### Deploy (1 hour)
```
Test new models
Deploy to 5% of traffic
Monitor performance
Gradually increase to 100% if successful
```

---

## Architecture

```
User Request
    ↓
Load Balancer
    ↓
API Servers (100 servers)
    ↓
Three parallel lookups:
    ├─ Redis (cached user data)
    ├─ FAISS (item search)
    └─ Feature Store (item/context data)
    ↓
Ranking Model
    ↓
Response
```

---

## Two-Stage Design

**Why two stages?**
- Can't score 10 million items fast enough
- Stage 1: Quick filter (10M → 500) using embeddings
- Stage 2: Precise ranking (500 → 20) using full model

**Stage 1 (Fast):**
- Method: Vector similarity
- Speed: 20ms
- Accuracy: 95%

**Stage 2 (Accurate):**
- Method: Machine learning model
- Speed: 15ms
- Accuracy: Very high

---

## Complete Request Path

```
1. Request arrives
2. Get user embedding from cache (1ms)
3. Search similar items in FAISS (20ms)
4. Fetch features in parallel (10ms)
5. Score items with LightGBM (15ms)
6. Apply diversity/freshness rules (5ms)
7. Format and send response (2ms)

Total: 57ms
```

## Training Pipeline (Daily)

```
Midnight:
    ↓
Collect yesterday's data
    ↓
Build features
    ↓
Train new models
    ↓
Test quality
    ↓
Deploy if better
    ↓
Next day: Use new models
```

---

That's it - the complete flow from user request to recommendations, and how models are updated daily.
