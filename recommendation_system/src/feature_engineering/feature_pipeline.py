"""
Feature Engineering Pipeline for Recommendation Systems

Handles mixed data types:
- Numerical: continuous and discrete
- Categorical: high cardinality features
- Text: NLP features from descriptions/reviews
- Sequential: user interaction history
- Temporal: time-based features

Staff Interview Topics:
- Feature engineering best practices
- Handling high-cardinality features
- Feature crosses and interactions
- Online/offline feature consistency
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, FloatType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    user_sequence_length: int = 50
    item_popularity_window_days: int = 30
    category_min_count: int = 10
    use_temporal_features: bool = True
    use_cross_features: bool = True


class UserFeatureEngineering:
    """
    User-level feature engineering

    Interview Topic: What makes a good user representation?
    """

    @staticmethod
    def compute_user_statistics(interactions_df: DataFrame) -> DataFrame:
        """
        Compute aggregate user behavior statistics

        Features:
        - Total interactions count
        - Interaction type distribution
        - Recency (days since last interaction)
        - Diversity (unique categories/brands)
        - Average session duration
        - Purchase frequency
        - Price sensitivity (avg price, std price)
        """
        # Interaction counts by type
        user_stats = interactions_df.groupBy("user_id").agg(
            F.count("*").alias("total_interactions"),
            F.countDistinct("item_id").alias("unique_items_interacted"),
            F.countDistinct("session_id").alias("total_sessions"),

            # Event type distribution
            F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
            F.sum(F.when(F.col("event_type") == "click", 1).otherwise(0)).alias("click_count"),
            F.sum(F.when(F.col("event_type") == "add_to_cart", 1).otherwise(0)).alias("cart_count"),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),

            # Recency
            F.max("timestamp").alias("last_interaction_timestamp"),
            F.min("timestamp").alias("first_interaction_timestamp"),

            # Duration and engagement
            F.avg("duration_seconds").alias("avg_session_duration"),
            F.sum("duration_seconds").alias("total_engagement_seconds"),

            # Price behavior
            F.avg("price").alias("avg_price_viewed"),
            F.stddev("price").alias("std_price_viewed"),
            F.avg(F.when(F.col("event_type") == "purchase", F.col("price"))).alias("avg_purchase_price"),
        )

        # Calculate recency (days since last interaction)
        user_stats = user_stats.withColumn(
            "recency_days",
            F.datediff(F.current_timestamp(), F.col("last_interaction_timestamp"))
        )

        # Calculate conversion rate
        user_stats = user_stats.withColumn(
            "conversion_rate",
            F.col("purchase_count") / (F.col("click_count") + 1)  # +1 to avoid division by zero
        )

        # Calculate engagement rate
        user_stats = user_stats.withColumn(
            "engagement_rate",
            F.col("click_count") / (F.col("view_count") + 1)
        )

        return user_stats

    @staticmethod
    def compute_user_preferences(
        interactions_df: DataFrame,
        item_metadata_df: DataFrame
    ) -> DataFrame:
        """
        Compute user preferences based on interaction history

        Features:
        - Favorite categories (top-3)
        - Favorite brands (top-3)
        - Category diversity (entropy)
        - Brand loyalty
        - Price range preference
        """
        # Join interactions with item metadata
        enriched = interactions_df.join(
            item_metadata_df,
            "item_id",
            "left"
        )

        # Category preferences (weighted by recency and event type)
        # More recent interactions and purchases get higher weight
        window_spec = Window.partitionBy("user_id", "category").orderBy(F.desc("timestamp"))

        category_prefs = enriched.groupBy("user_id", "category").agg(
            F.count("*").alias("category_interaction_count"),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("category_purchase_count"),
            F.max("timestamp").alias("last_category_interaction")
        )

        # Get top-3 categories per user
        window_rank = Window.partitionBy("user_id").orderBy(F.desc("category_interaction_count"))
        category_prefs = category_prefs.withColumn("category_rank", F.row_number().over(window_rank))

        top_categories = category_prefs.filter(F.col("category_rank") <= 3).groupBy("user_id").agg(
            F.collect_list("category").alias("top_categories"),
            F.collect_list("category_interaction_count").alias("top_category_counts")
        )

        # Brand preferences (similar approach)
        brand_prefs = enriched.groupBy("user_id", "brand").agg(
            F.count("*").alias("brand_interaction_count")
        )

        window_brand_rank = Window.partitionBy("user_id").orderBy(F.desc("brand_interaction_count"))
        brand_prefs = brand_prefs.withColumn("brand_rank", F.row_number().over(window_brand_rank))

        top_brands = brand_prefs.filter(F.col("brand_rank") <= 3).groupBy("user_id").agg(
            F.collect_list("brand").alias("top_brands")
        )

        # Combine preferences
        user_prefs = top_categories.join(top_brands, "user_id", "outer")

        return user_prefs

    @staticmethod
    def create_user_sequence_features(
        interactions_df: DataFrame,
        sequence_length: int = 50
    ) -> DataFrame:
        """
        Create sequential features (interaction history)

        Key Interview Point: Why sequences matter in recommendations
        - Capture temporal patterns
        - Learn from session context
        - Enable transformer-based models (SASRec, BERT4Rec)

        Returns:
            DataFrame with columns:
            - user_id
            - item_sequence: last N items interacted
            - event_sequence: corresponding event types
            - timestamp_sequence: interaction times
        """
        # Order interactions by timestamp
        window_spec = Window.partitionBy("user_id").orderBy(F.desc("timestamp"))

        # Get last N interactions per user
        sequences = interactions_df.withColumn(
            "row_num",
            F.row_number().over(window_spec)
        ).filter(
            F.col("row_num") <= sequence_length
        )

        # Collect into arrays
        user_sequences = sequences.groupBy("user_id").agg(
            F.collect_list("item_id").alias("item_sequence"),
            F.collect_list("event_type").alias("event_sequence"),
            F.collect_list(F.unix_timestamp("timestamp")).alias("timestamp_sequence"),
            F.count("*").alias("sequence_length")
        )

        return user_sequences


class ItemFeatureEngineering:
    """
    Item-level feature engineering

    Interview Topic: Content-based vs collaborative features
    """

    @staticmethod
    def compute_item_popularity(
        interactions_df: DataFrame,
        window_days: int = 30
    ) -> DataFrame:
        """
        Compute item popularity metrics

        Features:
        - Total views/clicks/purchases
        - CTR (click-through rate)
        - Conversion rate
        - Trending score (recent engagement vs historical)
        - Velocity (rate of popularity growth)
        """
        # Overall popularity
        item_stats = interactions_df.groupBy("item_id").agg(
            F.count("*").alias("total_interactions"),
            F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
            F.sum(F.when(F.col("event_type") == "click", 1).otherwise(0)).alias("click_count"),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),
            F.countDistinct("user_id").alias("unique_users")
        )

        # Calculate rates
        item_stats = item_stats.withColumn(
            "ctr",
            F.col("click_count") / (F.col("view_count") + 1)
        ).withColumn(
            "conversion_rate",
            F.col("purchase_count") / (F.col("click_count") + 1)
        )

        # Recent popularity (trending)
        recent_cutoff = F.date_sub(F.current_timestamp(), window_days)
        recent_interactions = interactions_df.filter(F.col("timestamp") >= recent_cutoff)

        recent_stats = recent_interactions.groupBy("item_id").agg(
            F.count("*").alias("recent_interactions"),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("recent_purchases")
        )

        # Join and compute trending score
        item_features = item_stats.join(recent_stats, "item_id", "left").fillna(0)

        item_features = item_features.withColumn(
            "trending_score",
            F.col("recent_interactions") / (F.col("total_interactions") + 1)
        )

        # Normalize popularity (log scale to handle skewed distribution)
        item_features = item_features.withColumn(
            "log_popularity",
            F.log1p(F.col("total_interactions"))
        )

        return item_features

    @staticmethod
    def process_text_features(item_metadata_df: DataFrame) -> DataFrame:
        """
        Process text features from item metadata

        Interview Topic: NLP in recommendation systems
        - TF-IDF vs embeddings
        - When to use what
        """
        from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

        # Combine title and description
        items_with_text = item_metadata_df.withColumn(
            "text_content",
            F.concat_ws(" ", F.col("title"), F.col("description"))
        )

        # Tokenization
        tokenizer = Tokenizer(inputCol="text_content", outputCol="words")
        tokenized = tokenizer.transform(items_with_text)

        # Remove stop words
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        filtered = remover.transform(tokenized)

        # TF-IDF (for traditional ML models)
        # Note: For deep learning, we'll use pre-trained embeddings (BERT/sentence-transformers)
        hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1024)
        featurized = hashing_tf.transform(filtered)

        idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
        idf_model = idf.fit(featurized)
        tfidf_df = idf_model.transform(featurized)

        return tfidf_df.select("item_id", "tfidf_features", "filtered_words")

    @staticmethod
    def encode_categorical_features(
        item_metadata_df: DataFrame,
        min_count: int = 10
    ) -> DataFrame:
        """
        Encode categorical features with frequency-based filtering

        Interview Topic: Handling high-cardinality categoricals
        - Frequency encoding
        - Target encoding
        - Embedding layers
        - Hashing trick
        """
        # For categories that appear less than min_count, replace with "OTHER"
        category_counts = item_metadata_df.groupBy("category").count()
        valid_categories = category_counts.filter(F.col("count") >= min_count).select("category")

        encoded = item_metadata_df.join(
            valid_categories,
            "category",
            "left"
        ).withColumn(
            "category_encoded",
            F.when(F.col("category").isNull(), "OTHER").otherwise(F.col("category"))
        )

        # Similar for brand
        brand_counts = item_metadata_df.groupBy("brand").count()
        valid_brands = brand_counts.filter(F.col("count") >= min_count).select("brand")

        encoded = encoded.join(
            valid_brands.withColumnRenamed("brand", "brand_valid"),
            F.col("brand") == F.col("brand_valid"),
            "left"
        ).withColumn(
            "brand_encoded",
            F.when(F.col("brand_valid").isNull(), "OTHER").otherwise(F.col("brand"))
        ).drop("brand_valid")

        return encoded


class ContextualFeatureEngineering:
    """
    Context-aware features (time, device, location)

    Interview Topic: Context in recommendations
    """

    @staticmethod
    def extract_temporal_features(df: DataFrame) -> DataFrame:
        """
        Extract time-based features

        Features:
        - Hour of day (cyclical encoding)
        - Day of week (cyclical encoding)
        - Is weekend
        - Is holiday season
        - Time since last interaction
        """
        df_temporal = df.withColumn("hour_of_day", F.hour("timestamp"))
        df_temporal = df_temporal.withColumn("day_of_week", F.dayofweek("timestamp"))
        df_temporal = df_temporal.withColumn("is_weekend",
                                              F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))

        # Cyclical encoding for hour (prevents discontinuity at midnight)
        df_temporal = df_temporal.withColumn(
            "hour_sin",
            F.sin(2 * np.pi * F.col("hour_of_day") / 24)
        ).withColumn(
            "hour_cos",
            F.cos(2 * np.pi * F.col("hour_of_day") / 24)
        )

        # Cyclical encoding for day of week
        df_temporal = df_temporal.withColumn(
            "day_sin",
            F.sin(2 * np.pi * F.col("day_of_week") / 7)
        ).withColumn(
            "day_cos",
            F.cos(2 * np.pi * F.col("day_of_week") / 7)
        )

        return df_temporal

    @staticmethod
    def create_cross_features(df: DataFrame) -> DataFrame:
        """
        Create feature crosses for interaction learning

        Interview Topic: When to use feature crosses vs deep learning
        - Explicit crosses: good for tree-based models
        - Deep learning: learns crosses implicitly
        """
        # User segment × Time cross
        df = df.withColumn(
            "user_segment_hour",
            F.concat_ws("_", F.col("account_type"), F.col("hour_of_day"))
        )

        # Device × Day cross
        df = df.withColumn(
            "device_day",
            F.concat_ws("_", F.col("device_type"), F.col("day_of_week"))
        )

        return df


class FeaturePipeline:
    """
    Main feature engineering orchestrator

    Interview Topic: End-to-end feature pipeline design
    """

    def __init__(self, config: FeatureConfig):
        self.config = config

    def create_features(
        self,
        interactions_df: DataFrame,
        user_profile_df: DataFrame,
        item_metadata_df: DataFrame
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Create all features for training

        Returns:
            (user_features, item_features, interaction_features)
        """
        logger.info("Starting feature engineering pipeline...")

        # User features
        logger.info("Computing user features...")
        user_stats = UserFeatureEngineering.compute_user_statistics(interactions_df)
        user_prefs = UserFeatureEngineering.compute_user_preferences(
            interactions_df,
            item_metadata_df
        )
        user_sequences = UserFeatureEngineering.create_user_sequence_features(
            interactions_df,
            self.config.user_sequence_length
        )

        user_features = user_stats.join(user_prefs, "user_id", "outer")
        user_features = user_features.join(user_sequences, "user_id", "outer")
        user_features = user_features.join(user_profile_df, "user_id", "left")

        # Item features
        logger.info("Computing item features...")
        item_popularity = ItemFeatureEngineering.compute_item_popularity(
            interactions_df,
            self.config.item_popularity_window_days
        )
        item_categoricals = ItemFeatureEngineering.encode_categorical_features(
            item_metadata_df,
            self.config.category_min_count
        )

        item_features = item_popularity.join(item_categoricals, "item_id", "left")

        # Interaction/Context features
        logger.info("Computing contextual features...")
        interaction_features = interactions_df
        if self.config.use_temporal_features:
            interaction_features = ContextualFeatureEngineering.extract_temporal_features(
                interaction_features
            )
        if self.config.use_cross_features:
            interaction_features = ContextualFeatureEngineering.create_cross_features(
                interaction_features
            )

        logger.info("Feature engineering complete!")

        return user_features, item_features, interaction_features


# Example usage
if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

    config = FeatureConfig(
        user_sequence_length=50,
        item_popularity_window_days=30,
        use_temporal_features=True
    )

    pipeline = FeaturePipeline(config)

    # Load data (from data_loader.py)
    # interactions_df = ...
    # user_profile_df = ...
    # item_metadata_df = ...

    # user_features, item_features, interaction_features = pipeline.create_features(
    #     interactions_df, user_profile_df, item_metadata_df
    # )
