"""
Data Loading and Validation Module

Handles data ingestion from various sources (S3, databases, streaming)
with validation, schema enforcement, and quality checks.

Staff Interview Topics:
- Data quality and validation
- Handling large-scale data
- Schema evolution
- Data versioning
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading"""
    source_path: str
    file_format: str = "parquet"
    date_partition_column: str = "event_date"
    validation_rules: Optional[Dict] = None
    sample_rate: float = 1.0


class DataSchema:
    """
    Define schemas for different data types

    Key Design Decision: Strongly typed schemas prevent downstream errors
    """

    @staticmethod
    def user_interactions_schema() -> StructType:
        """Schema for user-item interaction events"""
        return StructType([
            StructField("user_id", StringType(), False),
            StructField("item_id", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("event_type", StringType(), False),  # view, click, add_to_cart, purchase
            StructField("session_id", StringType(), True),
            StructField("device_type", StringType(), True),
            StructField("platform", StringType(), True),
            StructField("location_country", StringType(), True),
            StructField("price", FloatType(), True),
            StructField("discount_pct", FloatType(), True),
            StructField("duration_seconds", IntegerType(), True),
        ])

    @staticmethod
    def user_profile_schema() -> StructType:
        """Schema for user profile data"""
        return StructType([
            StructField("user_id", StringType(), False),
            StructField("created_at", TimestampType(), False),
            StructField("age_group", StringType(), True),  # 18-24, 25-34, etc.
            StructField("gender", StringType(), True),
            StructField("country", StringType(), True),
            StructField("language", StringType(), True),
            StructField("account_type", StringType(), True),  # free, premium, enterprise
            StructField("total_purchases", IntegerType(), True),
            StructField("lifetime_value", FloatType(), True),
            StructField("last_active_date", TimestampType(), True),
        ])

    @staticmethod
    def item_metadata_schema() -> StructType:
        """Schema for item/product metadata"""
        return StructType([
            StructField("item_id", StringType(), False),
            StructField("title", StringType(), True),
            StructField("description", StringType(), True),
            StructField("category", StringType(), True),
            StructField("subcategory", StringType(), True),
            StructField("brand", StringType(), True),
            StructField("price", FloatType(), True),
            StructField("created_at", TimestampType(), True),
            StructField("average_rating", FloatType(), True),
            StructField("num_ratings", IntegerType(), True),
            StructField("image_url", StringType(), True),
            StructField("tags", StringType(), True),  # JSON array stored as string
            StructField("is_available", StringType(), True),
        ])


class DataValidator:
    """
    Data quality validation

    Interview Topic: How to ensure data quality in production ML pipelines
    """

    @staticmethod
    def validate_interactions(df: DataFrame) -> Tuple[DataFrame, Dict]:
        """
        Validate interaction data quality

        Returns: (cleaned_df, validation_report)
        """
        initial_count = df.count()
        validation_report = {
            "initial_count": initial_count,
            "timestamp": datetime.now().isoformat()
        }

        # Remove null user_id or item_id (critical fields)
        df_clean = df.filter(
            (F.col("user_id").isNotNull()) &
            (F.col("item_id").isNotNull()) &
            (F.col("timestamp").isNotNull())
        )

        validation_report["null_critical_fields"] = initial_count - df_clean.count()

        # Remove duplicates (same user, item, timestamp)
        df_clean = df_clean.dropDuplicates(["user_id", "item_id", "timestamp"])
        validation_report["duplicates_removed"] = initial_count - df_clean.count()

        # Filter invalid timestamps (future dates or too old)
        current_date = F.current_timestamp()
        df_clean = df_clean.filter(
            (F.col("timestamp") <= current_date) &
            (F.col("timestamp") >= F.date_sub(current_date, 730))  # 2 years window
        )

        # Validate event types
        valid_event_types = ["view", "click", "add_to_cart", "purchase", "favorite"]
        df_clean = df_clean.filter(F.col("event_type").isin(valid_event_types))

        validation_report["final_count"] = df_clean.count()
        validation_report["data_quality_rate"] = validation_report["final_count"] / initial_count

        logger.info(f"Data validation complete: {validation_report}")

        return df_clean, validation_report

    @staticmethod
    def detect_anomalies(df: DataFrame) -> Dict:
        """
        Detect statistical anomalies in data

        Interview Topic: Monitoring data drift and anomalies
        """
        anomalies = {}

        # Check for unusual spike in events
        daily_counts = df.groupBy(F.to_date("timestamp").alias("date")).count()
        stats = daily_counts.select(
            F.mean("count").alias("mean"),
            F.stddev("count").alias("std")
        ).collect()[0]

        anomalies["daily_event_mean"] = stats["mean"]
        anomalies["daily_event_std"] = stats["std"]

        # Check for bot-like behavior (too many events from single user)
        user_event_counts = df.groupBy("user_id").count()
        max_events = user_event_counts.agg(F.max("count")).collect()[0][0]

        if max_events > 1000:  # Threshold for suspicious activity
            anomalies["potential_bot_users"] = user_event_counts.filter(
                F.col("count") > 1000
            ).count()

        return anomalies


class DataLoader:
    """
    Main data loading orchestrator

    Interview Topics:
    - Handling large-scale data (billions of rows)
    - Incremental loading strategies
    - Partitioning strategies for performance
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load_interactions(
        self,
        path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = True
    ) -> DataFrame:
        """
        Load user-item interaction data with optional date filtering

        Args:
            path: S3/GCS path to parquet files
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            validate: Run validation checks

        Returns:
            Spark DataFrame with interactions
        """
        logger.info(f"Loading interactions from {path}")

        df = self.spark.read.parquet(path)

        # Apply date filtering if provided
        if start_date:
            df = df.filter(F.col("timestamp") >= start_date)
        if end_date:
            df = df.filter(F.col("timestamp") <= end_date)

        if validate:
            df, report = DataValidator.validate_interactions(df)

            # Log validation metrics for monitoring
            logger.info(f"Validation report: {report}")

            # Check data quality threshold
            if report["data_quality_rate"] < 0.95:
                logger.warning(
                    f"Data quality below threshold: {report['data_quality_rate']:.2%}"
                )

        return df

    def load_user_profiles(self, path: str) -> DataFrame:
        """Load user profile data (typically smaller, full load)"""
        logger.info(f"Loading user profiles from {path}")
        return self.spark.read.parquet(path)

    def load_item_metadata(self, path: str) -> DataFrame:
        """Load item metadata (catalog data)"""
        logger.info(f"Loading item metadata from {path}")
        return self.spark.read.parquet(path)

    def create_train_test_split(
        self,
        df: DataFrame,
        test_days: int = 7,
        val_days: int = 7
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Time-based train/validation/test split

        Key Interview Point: Why time-based split is critical for recommender systems
        - Prevents data leakage
        - Simulates production scenario (predicting future from past)
        - Accounts for temporal patterns
        """
        max_date = df.agg(F.max("timestamp")).collect()[0][0]

        test_start = F.date_sub(F.lit(max_date), test_days)
        val_start = F.date_sub(F.lit(max_date), test_days + val_days)

        train_df = df.filter(F.col("timestamp") < val_start)
        val_df = df.filter(
            (F.col("timestamp") >= val_start) &
            (F.col("timestamp") < test_start)
        )
        test_df = df.filter(F.col("timestamp") >= test_start)

        logger.info(f"Train: {train_df.count()}, Val: {val_df.count()}, Test: {test_df.count()}")

        return train_df, val_df, test_df


# Example Usage
if __name__ == "__main__":
    # Initialize Spark with appropriate configurations for large-scale processing
    spark = (SparkSession.builder
             .appName("RecommendationDataPipeline")
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
             .config("spark.sql.files.maxPartitionBytes", "128MB")
             .getOrCreate())

    loader = DataLoader(spark)

    # Load and validate data
    interactions = loader.load_interactions(
        path="s3://my-bucket/interactions/",
        start_date="2024-01-01",
        validate=True
    )

    # Create temporal splits
    train, val, test = loader.create_train_test_split(interactions)

    # Save processed data
    train.write.mode("overwrite").parquet("s3://my-bucket/processed/train/")
    val.write.mode("overwrite").parquet("s3://my-bucket/processed/val/")
    test.write.mode("overwrite").parquet("s3://my-bucket/processed/test/")
