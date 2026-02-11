"""
Production Recommendation Serving System

Two-Stage Retrieval Architecture:
1. Candidate Generation: Fast retrieval from millions (ANN search)
2. Ranking: Precise scoring of hundreds (complex model)

Staff Interview Topics:
- Low-latency serving (< 100ms p99)
- Caching strategies
- A/B testing framework
- Online feature computation
- Model serving infrastructure
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import redis
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServingConfig:
    """Configuration for serving system"""
    # Retrieval stage
    num_candidates: int = 500
    ann_index_type: str = "IVF"

    # Ranking stage
    num_final_items: int = 50
    ranking_batch_size: int = 128

    # Caching
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl_seconds: int = 3600

    # Performance
    max_latency_ms: float = 100.0
    num_threads: int = 4

    # Business rules
    enable_diversity: bool = True
    diversity_window: int = 5
    max_items_per_category: int = 3
    enable_freshness: bool = True
    freshness_weight: float = 0.1


# Request/Response Models
class RecommendationRequest(BaseModel):
    """API request schema"""
    user_id: str
    context: Optional[Dict] = None  # device, location, time, etc.
    num_items: int = 20
    filters: Optional[Dict] = None  # category, price_range, etc.
    experiment_id: Optional[str] = None  # for A/B testing


class RecommendationResponse(BaseModel):
    """API response schema"""
    user_id: str
    items: List[Dict]  # [{item_id, score, rank, metadata}, ...]
    request_id: str
    latency_ms: float
    stage_latencies: Dict[str, float]
    model_version: str


class CandidateGenerator:
    """
    Stage 1: Fast candidate generation using embeddings + ANN

    Interview Topic: Why two-stage architecture?
    - Can't run complex model on millions of items (too slow)
    - ANN search: O(log n) vs O(n) for brute force
    - Sub-linear scaling is critical for real-time serving
    """

    def __init__(
        self,
        embedding_dim: int,
        config: ServingConfig,
        redis_client: Optional[redis.Redis] = None
    ):
        self.embedding_dim = embedding_dim
        self.config = config
        self.redis = redis_client

        # FAISS index for ANN search
        self.item_index = None
        self.item_id_mapping = None  # index -> item_id

        # User embedding model (loaded separately)
        self.user_encoder = None

    def load_item_index(self, index_path: str, mapping_path: str):
        """
        Load pre-built FAISS index and ID mapping

        Interview Point: Pre-computation strategy
        - Item embeddings are static (updated daily/hourly)
        - User embeddings computed on-the-fly (dynamic)
        - Enables fast serving without re-encoding items
        """
        self.item_index = faiss.read_index(index_path)
        self.item_id_mapping = np.load(mapping_path)
        logger.info(f"Loaded index with {self.item_index.ntotal} items")

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """
        Retrieve user embedding from cache or compute

        Interview Topic: Online feature serving
        - Option 1: Pre-compute and cache (fast, stale)
        - Option 2: Compute on-the-fly (slow, fresh)
        - Option 3: Hybrid (cache + fallback)
        """
        if self.redis:
            # Try cache first
            cache_key = f"user_emb:{user_id}"
            cached = self.redis.get(cache_key)

            if cached:
                return np.frombuffer(cached, dtype=np.float32)

        # Fallback: compute from user features
        # In production, this would fetch from feature store
        user_features = self._fetch_user_features(user_id)
        if user_features is None:
            return None

        # Encode using user tower model
        with torch.no_grad():
            user_emb = self.user_encoder(user_features).cpu().numpy()

        # Cache for future requests
        if self.redis:
            self.redis.setex(
                cache_key,
                self.config.cache_ttl_seconds,
                user_emb.tobytes()
            )

        return user_emb

    def _fetch_user_features(self, user_id: str) -> Optional[torch.Tensor]:
        """
        Fetch user features from feature store

        Interview Topic: Feature store architecture
        - Online vs offline features
        - Point-in-time correctness
        - Low-latency requirements (< 10ms)

        Popular solutions: Feast, Tecton, AWS SageMaker Feature Store
        """
        # Placeholder: In production, query feature store
        # Example: Feast
        # features = feast_client.get_online_features(
        #     features=['user_age', 'user_ltv', 'user_category_prefs'],
        #     entity_rows=[{'user_id': user_id}]
        # ).to_dict()

        # Mock implementation
        return None

    def retrieve_candidates(
        self,
        user_id: str,
        num_candidates: int,
        filters: Optional[Dict] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve candidate items using ANN search

        Returns:
            (item_ids, scores)

        Interview Topic: Filtering in ANN search
        - Pre-filtering: Filter before search (may miss results)
        - Post-filtering: Search then filter (may be slow)
        - Hybrid: Over-retrieve then filter
        """
        start_time = time.time()

        # Get user embedding
        user_emb = self.get_user_embedding(user_id)
        if user_emb is None:
            # Cold start fallback: popular items
            return self._get_popular_items(num_candidates), [1.0] * num_candidates

        # Normalize for cosine similarity
        user_emb = user_emb / np.linalg.norm(user_emb)

        # Search
        # Over-retrieve to account for filtering
        retrieve_k = num_candidates * 3 if filters else num_candidates

        distances, indices = self.item_index.search(
            user_emb.reshape(1, -1).astype('float32'),
            retrieve_k
        )

        # Map indices to item IDs
        item_ids = [self.item_id_mapping[idx] for idx in indices[0]]
        scores = distances[0].tolist()

        # Apply filters if needed
        if filters:
            filtered = []
            filtered_scores = []

            for item_id, score in zip(item_ids, scores):
                if self._check_filters(item_id, filters):
                    filtered.append(item_id)
                    filtered_scores.append(score)

                if len(filtered) >= num_candidates:
                    break

            item_ids = filtered
            scores = filtered_scores

        latency = (time.time() - start_time) * 1000
        logger.info(f"Candidate generation: {len(item_ids)} items in {latency:.2f}ms")

        return item_ids[:num_candidates], scores[:num_candidates]

    def _get_popular_items(self, k: int) -> List[str]:
        """Fallback: return popular items for cold start"""
        # In production, maintain a cached list of trending/popular items
        return [f"item_{i}" for i in range(k)]

    def _check_filters(self, item_id: str, filters: Dict) -> bool:
        """Apply business logic filters"""
        # Placeholder: check item metadata against filters
        return True


class RankingService:
    """
    Stage 2: Re-rank candidates using complex model

    Interview Topic: Ranking model selection
    - LightGBM: Fast, < 10ms for 500 items
    - Neural: Slower, 20-50ms but better accuracy
    - Hybrid: Neural for top candidates, GBDT for rest
    """

    def __init__(self, config: ServingConfig):
        self.config = config
        self.model = None
        self.model_version = "v1.0"

    def load_model(self, model_path: str):
        """Load ranking model"""
        # Example for LightGBM
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=model_path)
        logger.info(f"Loaded ranking model from {model_path}")

    def rank_candidates(
        self,
        user_id: str,
        item_ids: List[str],
        context: Optional[Dict] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Score and rank candidate items

        Interview Topic: Feature computation latency
        - Pre-computed features: Fast but stale
        - Real-time features: Fresh but slow
        - Critical path optimization
        """
        start_time = time.time()

        # Fetch features for all user-item pairs
        features = self._create_ranking_features(user_id, item_ids, context)

        # Batch prediction
        scores = self.model.predict(features)

        # Sort by score
        ranked_indices = np.argsort(-scores)
        ranked_item_ids = [item_ids[i] for i in ranked_indices]
        ranked_scores = [scores[i] for i in ranked_indices]

        latency = (time.time() - start_time) * 1000
        logger.info(f"Ranking: {len(item_ids)} items in {latency:.2f}ms")

        return ranked_item_ids, ranked_scores

    def _create_ranking_features(
        self,
        user_id: str,
        item_ids: List[str],
        context: Optional[Dict]
    ) -> np.ndarray:
        """
        Create feature vectors for ranking

        Features include:
        - User features (demographics, behavior, preferences)
        - Item features (metadata, popularity, quality)
        - User-item interaction features (affinity, similarity)
        - Context features (time, device, location)
        - Cross features (user_segment × time, etc.)

        Interview Topic: Feature parallelization
        - Use ThreadPoolExecutor for parallel feature fetching
        - Critical for meeting latency SLA
        """
        num_items = len(item_ids)

        # Placeholder: In production, fetch from feature store in parallel
        # Use ThreadPoolExecutor to parallelize
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            user_features = executor.submit(self._get_user_features, user_id)
            item_features = executor.submit(self._get_item_features, item_ids)
            context_features = executor.submit(self._get_context_features, context)

            # Wait for all
            user_feat = user_features.result()
            item_feat = item_features.result()
            ctx_feat = context_features.result()

        # Combine features
        # Shape: [num_items, total_features]
        features = np.zeros((num_items, 100))  # Placeholder

        return features

    def _get_user_features(self, user_id: str) -> np.ndarray:
        """Fetch user features from feature store"""
        return np.random.randn(50)  # Placeholder

    def _get_item_features(self, item_ids: List[str]) -> np.ndarray:
        """Fetch item features from feature store"""
        return np.random.randn(len(item_ids), 30)  # Placeholder

    def _get_context_features(self, context: Optional[Dict]) -> np.ndarray:
        """Extract context features"""
        return np.random.randn(20)  # Placeholder


class BusinessLogicLayer:
    """
    Post-processing: Apply business rules

    Interview Topic: Why business logic post-processing?
    - Model optimizes for engagement, not necessarily business goals
    - Need diversity, freshness, fairness
    - Regulatory requirements (e.g., filter certain content)
    """

    def __init__(self, config: ServingConfig):
        self.config = config

    def apply_rules(
        self,
        item_ids: List[str],
        scores: List[float],
        user_id: str
    ) -> Tuple[List[str], List[float]]:
        """
        Apply business logic rules

        Rules:
        1. Diversity: Don't show too many items from same category
        2. Freshness: Boost recently added items
        3. Deduplication: Remove recently shown items
        4. Fairness: Ensure representation from various creators
        """
        # Deduplication
        item_ids, scores = self._deduplicate(user_id, item_ids, scores)

        # Diversity
        if self.config.enable_diversity:
            item_ids, scores = self._enforce_diversity(item_ids, scores)

        # Freshness boost
        if self.config.enable_freshness:
            scores = self._apply_freshness_boost(item_ids, scores)

        # Re-sort after adjustments
        sorted_indices = np.argsort(-np.array(scores))
        item_ids = [item_ids[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]

        return item_ids, scores

    def _deduplicate(
        self,
        user_id: str,
        item_ids: List[str],
        scores: List[float]
    ) -> Tuple[List[str], List[float]]:
        """Remove items user has recently interacted with"""
        # Fetch recent history from Redis/database
        # For now, simple placeholder
        recently_shown = set()  # Would fetch from cache

        filtered_items = []
        filtered_scores = []

        for item_id, score in zip(item_ids, scores):
            if item_id not in recently_shown:
                filtered_items.append(item_id)
                filtered_scores.append(score)

        return filtered_items, filtered_scores

    def _enforce_diversity(
        self,
        item_ids: List[str],
        scores: List[float]
    ) -> Tuple[List[str], List[float]]:
        """
        Ensure diversity across categories

        Interview Topic: Diversity algorithms
        - MMR (Maximal Marginal Relevance)
        - Sliding window approach
        - DPP (Determinantal Point Processes)
        """
        # Sliding window approach
        category_counts = {}
        diverse_items = []
        diverse_scores = []

        for item_id, score in zip(item_ids, scores):
            category = self._get_item_category(item_id)

            count = category_counts.get(category, 0)
            if count < self.config.max_items_per_category:
                diverse_items.append(item_id)
                diverse_scores.append(score)
                category_counts[category] = count + 1

        return diverse_items, diverse_scores

    def _apply_freshness_boost(
        self,
        item_ids: List[str],
        scores: List[float]
    ) -> List[float]:
        """Boost recently added items"""
        boosted_scores = []

        for item_id, score in zip(item_ids, scores):
            age_days = self._get_item_age_days(item_id)

            # Exponential decay: boost = e^(-age/30)
            freshness_boost = np.exp(-age_days / 30)
            boosted_score = score * (1 + self.config.freshness_weight * freshness_boost)
            boosted_scores.append(boosted_score)

        return boosted_scores

    def _get_item_category(self, item_id: str) -> str:
        """Get item category from metadata"""
        return "default"  # Placeholder

    def _get_item_age_days(self, item_id: str) -> int:
        """Get item age in days"""
        return 1  # Placeholder


class RecommendationService:
    """
    Main recommendation service orchestrator

    Interview Topic: End-to-end serving architecture
    """

    def __init__(self, config: ServingConfig):
        self.config = config

        # Initialize components
        redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=False
        )

        self.candidate_generator = CandidateGenerator(
            embedding_dim=128,
            config=config,
            redis_client=redis_client
        )
        self.ranking_service = RankingService(config)
        self.business_logic = BusinessLogicLayer(config)

    def recommend(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """
        Generate recommendations

        Interview Topic: Latency breakdown
        - Candidate generation: 20-40ms
        - Feature fetching: 10-20ms
        - Ranking: 10-30ms
        - Business logic: 5-10ms
        - Total: < 100ms p99
        """
        start_time = time.time()
        stage_latencies = {}

        # Stage 1: Candidate Generation
        stage_start = time.time()
        candidate_ids, candidate_scores = self.candidate_generator.retrieve_candidates(
            request.user_id,
            self.config.num_candidates,
            request.filters
        )
        stage_latencies['candidate_generation'] = (time.time() - stage_start) * 1000

        # Stage 2: Ranking
        stage_start = time.time()
        ranked_ids, ranked_scores = self.ranking_service.rank_candidates(
            request.user_id,
            candidate_ids,
            request.context
        )
        stage_latencies['ranking'] = (time.time() - stage_start) * 1000

        # Stage 3: Business Logic
        stage_start = time.time()
        final_ids, final_scores = self.business_logic.apply_rules(
            ranked_ids,
            ranked_scores,
            request.user_id
        )
        stage_latencies['business_logic'] = (time.time() - stage_start) * 1000

        # Prepare response
        items = []
        for rank, (item_id, score) in enumerate(
            zip(final_ids[:request.num_items], final_scores[:request.num_items])
        ):
            items.append({
                'item_id': item_id,
                'score': float(score),
                'rank': rank + 1,
                'metadata': {}  # Can add item metadata here
            })

        total_latency = (time.time() - start_time) * 1000

        # Check latency SLA
        if total_latency > self.config.max_latency_ms:
            logger.warning(f"Latency SLA violated: {total_latency:.2f}ms")

        return RecommendationResponse(
            user_id=request.user_id,
            items=items,
            request_id=f"{int(time.time())}_{request.user_id}",
            latency_ms=total_latency,
            stage_latencies=stage_latencies,
            model_version=self.ranking_service.model_version
        )


# FastAPI application
app = FastAPI(title="Recommendation Service")

# Initialize service
config = ServingConfig()
rec_service = RecommendationService(config)


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized recommendations

    Interview Topic: API design for ML services
    - Request/response schema
    - Error handling
    - Latency monitoring
    - A/B testing integration
    """
    try:
        response = rec_service.recommend(request)
        return response
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": rec_service.ranking_service.model_version}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
