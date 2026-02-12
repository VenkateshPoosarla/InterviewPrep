"""
Feature Engineering for Propensity Modeling

Creates features from raw customer data:
- RFM (Recency, Frequency, Monetary) features
- Engagement scores
- Behavioral patterns
- Interaction features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime


class PropensityFeatureEngineering:
    """Feature engineering pipeline for propensity models"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []

    def create_features(self, df):
        """
        Create all features from raw customer data

        Returns:
            df_features: DataFrame with engineered features
            feature_names: List of feature column names
        """
        print("Creating features...")

        df_features = df.copy()

        # 1. RFM Features
        df_features = self._create_rfm_features(df_features)

        # 2. Engagement Features
        df_features = self._create_engagement_features(df_features)

        # 3. Behavioral Features
        df_features = self._create_behavioral_features(df_features)

        # 4. Interaction Features
        df_features = self._create_interaction_features(df_features)

        # 5. Binned Features
        df_features = self._create_binned_features(df_features)

        # 6. Encode categorical features
        df_features = self._encode_categorical(df_features)

        # Store feature names (exclude ID and target)
        self.feature_names = [col for col in df_features.columns
                             if col not in ['customer_id', 'converted']]

        print(f"✓ Created {len(self.feature_names)} features")

        return df_features

    def _create_rfm_features(self, df):
        """
        Create RFM (Recency, Frequency, Monetary) features

        RFM is a classic framework for customer segmentation:
        - Recency: How recently did customer purchase?
        - Frequency: How often do they purchase?
        - Monetary: How much do they spend?
        """
        # Recency score (inverse of days since last purchase)
        df['recency_score'] = 1 / (df['days_since_last_purchase'] + 1)

        # Frequency score (normalized purchase frequency)
        df['frequency_score'] = df['purchase_frequency']

        # Monetary score (normalized total revenue)
        df['monetary_score'] = df['total_revenue']

        # RFM composite score
        df['rfm_score'] = (
            df['recency_score'] * 0.4 +
            df['frequency_score'] * 0.3 +
            df['monetary_score'] * 0.3
        )

        # RFM segment (High, Medium, Low)
        df['rfm_segment'] = pd.qcut(
            df['rfm_score'],
            q=3,
            labels=['Low', 'Medium', 'High']
        )

        return df

    def _create_engagement_features(self, df):
        """Create engagement-related features"""

        # Email engagement score
        df['email_engagement_score'] = (
            df['email_open_rate'] * 0.6 +
            df['email_click_rate'] * 0.4
        )

        # Web engagement score
        max_visits = df['website_visits_30d'].max() + 1
        max_duration = df['avg_session_duration_min'].max() + 1

        df['web_engagement_score'] = (
            (df['website_visits_30d'] / max_visits) * 0.5 +
            (df['avg_session_duration_min'] / max_duration) * 0.3 +
            (df['pages_per_session'] / df['pages_per_session'].max()) * 0.2
        )

        # Overall engagement score
        df['overall_engagement'] = (
            df['email_engagement_score'] * 0.4 +
            df['web_engagement_score'] * 0.4 +
            df['has_mobile_app'] * 0.1 +
            df['social_media_follower'] * 0.1
        )

        # Engagement level category
        df['engagement_level'] = pd.cut(
            df['overall_engagement'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=['Low', 'Medium', 'High']
        )

        return df

    def _create_behavioral_features(self, df):
        """Create behavioral pattern features"""

        # Customer lifecycle stage
        df['lifecycle_stage'] = pd.cut(
            df['tenure_months'],
            bins=[0, 3, 12, 24, 120],
            labels=['New', 'Growing', 'Mature', 'Loyal']
        )

        # Purchase velocity (purchases per month)
        df['purchase_velocity'] = np.where(
            df['tenure_months'] > 0,
            df['total_purchases'] / df['tenure_months'],
            0
        )

        # Value tier based on AOV
        df['value_tier'] = pd.qcut(
            df['avg_order_value'],
            q=4,
            labels=['Bronze', 'Silver', 'Gold', 'Platinum']
        )

        # Discount dependency
        df['discount_dependent'] = (df['purchases_with_discount_pct'] > 0.7).astype(int)

        # Browse to purchase ratio
        df['browse_to_purchase_ratio'] = np.where(
            df['product_views_30d'] > 0,
            df['total_purchases'] / df['product_views_30d'],
            0
        )

        # Cart abandonment flag
        df['high_cart_abandonment'] = (df['cart_abandonment_rate'] > 0.5).astype(int)

        # Wishlist engagement
        df['wishlist_engaged'] = (df['wishlist_items'] > 0).astype(int)

        return df

    def _create_interaction_features(self, df):
        """Create interaction features (feature crosses)"""

        # Engagement × Recency
        df['engaged_and_recent'] = (
            df['overall_engagement'] *
            df['recency_score']
        )

        # Value × Frequency
        df['high_value_frequent'] = (
            df['avg_order_value'] *
            df['purchase_frequency']
        )

        # Email engagement × Web engagement
        df['omnichannel_engaged'] = (
            df['email_engagement_score'] *
            df['web_engagement_score']
        )

        # App user × High engagement
        df['app_power_user'] = (
            df['has_mobile_app'] *
            (df['overall_engagement'] > 0.7).astype(int)
        )

        # Tenure × Purchase frequency
        df['loyal_frequent'] = (
            (df['tenure_months'] > 12).astype(int) *
            (df['purchase_frequency'] > df['purchase_frequency'].median()).astype(int)
        )

        return df

    def _create_binned_features(self, df):
        """Create binned versions of continuous features"""

        # Age groups
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )

        # Tenure bins
        df['tenure_bin'] = pd.cut(
            df['tenure_months'],
            bins=[0, 6, 12, 24, 48, 120],
            labels=['0-6mo', '6-12mo', '1-2yr', '2-4yr', '4yr+']
        )

        # Revenue bins
        df['revenue_bin'] = pd.qcut(
            df['total_revenue'],
            q=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            duplicates='drop'
        )

        return df

    def _encode_categorical(self, df):
        """Encode categorical features"""

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != 'customer_id']

        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))

        return df

    def scale_features(self, df, feature_cols=None):
        """Scale numerical features"""

        if feature_cols is None:
            feature_cols = self.feature_names

        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns

        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df


def create_propensity_features(df, scale=True):
    """
    Main function to create all propensity features

    Args:
        df: Raw customer data
        scale: Whether to scale numerical features

    Returns:
        df_features: DataFrame with all engineered features
        feature_names: List of feature column names
    """
    engineer = PropensityFeatureEngineering()
    df_features = engineer.create_features(df)

    if scale:
        df_features = engineer.scale_features(df_features)

    return df_features, engineer.feature_names


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering...")

    # Load sample data
    df = pd.read_parquet('data/raw/customer_data.parquet')
    print(f"Loaded {len(df):,} customers")

    # Create features
    df_features, feature_names = create_propensity_features(df)

    print(f"\nFeatures created: {len(feature_names)}")
    print(f"Data shape: {df_features.shape}")

    # Save processed data
    df_features.to_parquet('data/processed/customer_features.parquet', index=False)
    print("\n✓ Saved processed features")
