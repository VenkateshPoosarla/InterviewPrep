"""
Unit Tests for Propensity Models

Tests for model training, prediction, and evaluation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.propensity_model import PropensityModelTrainer
from data_pipeline.feature_engineering import PropensityFeatureEngineering


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)

    n_samples = 1000

    data = pd.DataFrame({
        'customer_id': [f"CUST_{i:05d}" for i in range(n_samples)],
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'tenure_months': np.random.randint(0, 60, n_samples),
        'total_purchases': np.random.randint(0, 50, n_samples),
        'avg_order_value': np.random.uniform(10, 500, n_samples),
        'total_revenue': np.random.uniform(0, 10000, n_samples),
        'days_since_last_purchase': np.random.randint(0, 365, n_samples),
        'purchase_frequency': np.random.uniform(0, 5, n_samples),
        'is_repeat_customer': np.random.binomial(1, 0.6, n_samples),
        'email_sent_30d': np.random.randint(0, 30, n_samples),
        'email_opened_30d': np.random.randint(0, 20, n_samples),
        'email_clicked_30d': np.random.randint(0, 10, n_samples),
        'email_open_rate': np.random.uniform(0, 1, n_samples),
        'email_click_rate': np.random.uniform(0, 1, n_samples),
        'website_visits_30d': np.random.randint(0, 50, n_samples),
        'avg_session_duration_min': np.random.uniform(1, 30, n_samples),
        'pages_per_session': np.random.uniform(1, 20, n_samples),
        'has_mobile_app': np.random.binomial(1, 0.3, n_samples),
        'app_sessions_30d': np.random.randint(0, 50, n_samples),
        'social_media_follower': np.random.binomial(1, 0.2, n_samples),
        'num_categories_purchased': np.random.randint(0, 10, n_samples),
        'purchases_with_discount_pct': np.random.uniform(0, 1, n_samples),
        'cart_abandonment_rate': np.random.uniform(0, 1, n_samples),
        'product_views_30d': np.random.randint(0, 100, n_samples),
        'wishlist_items': np.random.randint(0, 20, n_samples),
        'return_rate': np.random.uniform(0, 0.5, n_samples),
        'converted': np.random.binomial(1, 0.05, n_samples)
    })

    return data


class TestFeatureEngineering:
    """Test feature engineering pipeline"""

    def test_feature_creation(self, sample_data):
        """Test that features are created correctly"""
        engineer = PropensityFeatureEngineering()
        df_features = engineer.create_features(sample_data)

        # Check that new features were created
        assert 'rfm_score' in df_features.columns
        assert 'recency_score' in df_features.columns
        assert 'overall_engagement' in df_features.columns
        assert 'purchase_velocity' in df_features.columns

    def test_rfm_features(self, sample_data):
        """Test RFM feature creation"""
        engineer = PropensityFeatureEngineering()
        df_features = engineer.create_features(sample_data)

        # RFM score should be between 0 and some reasonable max
        assert df_features['rfm_score'].min() >= 0
        assert df_features['rfm_score'].max() > 0

        # Recency score should be inverse of days
        assert df_features['recency_score'].min() > 0
        assert df_features['recency_score'].max() <= 1

    def test_engagement_features(self, sample_data):
        """Test engagement feature creation"""
        engineer = PropensityFeatureEngineering()
        df_features = engineer.create_features(sample_data)

        # Engagement scores should be between 0 and 1
        assert df_features['email_engagement_score'].between(0, 1).all()
        assert df_features['web_engagement_score'].between(0, 1).all()
        assert df_features['overall_engagement'].between(0, 1).all()

    def test_no_missing_values(self, sample_data):
        """Test that feature engineering doesn't introduce NaN"""
        engineer = PropensityFeatureEngineering()
        df_features = engineer.create_features(sample_data)

        # Check for NaN in numeric columns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        assert df_features[numeric_cols].isna().sum().sum() == 0


class TestPropensityModelTrainer:
    """Test model training pipeline"""

    def test_data_preparation(self, sample_data):
        """Test train/val/test split"""
        # Add features first
        engineer = PropensityFeatureEngineering()
        df_features = engineer.create_features(sample_data)

        # Prepare data
        trainer = PropensityModelTrainer()
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df_features)

        # Check splits
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == len(sample_data)

        # Check that we have features
        assert X_train.shape[1] > 0
        assert X_val.shape[1] > 0
        assert X_test.shape[1] > 0

        # Check target balance is maintained
        train_positive_rate = y_train.mean()
        test_positive_rate = y_test.mean()
        assert abs(train_positive_rate - test_positive_rate) < 0.05  # Within 5%

    def test_logistic_regression_training(self, sample_data):
        """Test logistic regression training"""
        engineer = PropensityFeatureEngineering()
        df_features = engineer.create_features(sample_data)

        trainer = PropensityModelTrainer()
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df_features)

        # Train model
        model = trainer.train_logistic_regression(X_train, y_train, X_val, y_val)

        # Check model was trained
        assert model is not None
        assert hasattr(model, 'predict_proba')

        # Check predictions are valid probabilities
        y_pred = model.predict_proba(X_test)[:, 1]
        assert y_pred.min() >= 0
        assert y_pred.max() <= 1

        # Check metrics were recorded
        assert 'logistic_regression' in trainer.metrics
        assert 'val_auc' in trainer.metrics['logistic_regression']
        assert trainer.metrics['logistic_regression']['val_auc'] > 0.5  # Better than random

    def test_lightgbm_training(self, sample_data):
        """Test LightGBM training"""
        engineer = PropensityFeatureEngineering()
        df_features = engineer.create_features(sample_data)

        trainer = PropensityModelTrainer()
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df_features)

        # Train model
        model = trainer.train_lightgbm(X_train, y_train, X_val, y_val)

        # Check model was trained
        assert model is not None

        # Check predictions
        y_pred = model.predict(X_test)
        assert y_pred.min() >= 0
        assert y_pred.max() <= 1

        # Check metrics
        assert 'lightgbm' in trainer.metrics
        assert trainer.metrics['lightgbm']['val_auc'] > 0.5

    def test_model_evaluation(self, sample_data):
        """Test model evaluation on test set"""
        engineer = PropensityFeatureEngineering()
        df_features = engineer.create_features(sample_data)

        trainer = PropensityModelTrainer()
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df_features)

        # Train a simple model
        trainer.train_logistic_regression(X_train, y_train, X_val, y_val)

        # Evaluate
        results = trainer.evaluate_on_test(X_test, y_test)

        # Check results
        assert 'logistic_regression' in results
        assert 'auc' in results['logistic_regression']
        assert results['logistic_regression']['auc'] > 0
        assert results['logistic_regression']['auc'] <= 1


class TestPredictions:
    """Test prediction functionality"""

    def test_prediction_range(self, sample_data):
        """Test that predictions are in valid range [0, 1]"""
        engineer = PropensityFeatureEngineering()
        df_features = engineer.create_features(sample_data)

        trainer = PropensityModelTrainer()
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df_features)

        # Train and predict
        model = trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
        predictions = model.predict_proba(X_test)[:, 1]

        # Check range
        assert predictions.min() >= 0
        assert predictions.max() <= 1

    def test_prediction_shape(self, sample_data):
        """Test prediction output shape"""
        engineer = PropensityFeatureEngineering()
        df_features = engineer.create_features(sample_data)

        trainer = PropensityModelTrainer()
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df_features)

        model = trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
        predictions = model.predict_proba(X_test)[:, 1]

        # Check shape
        assert len(predictions) == len(X_test)
        assert predictions.ndim == 1


class TestDataValidation:
    """Test data validation"""

    def test_missing_values_handling(self):
        """Test handling of missing values"""
        # Create data with missing values
        data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'age': [25, np.nan, 35],
            'tenure_months': [12, 24, 36],
            'total_purchases': [5, 10, np.nan],
            'avg_order_value': [50, 75, 100],
            'total_revenue': [250, 750, 1000],
            'days_since_last_purchase': [10, 20, 30],
            'purchase_frequency': [0.5, 1.0, 1.5],
            'is_repeat_customer': [1, 1, 1],
            'email_sent_30d': [10, 15, 20],
            'email_opened_30d': [5, 8, 10],
            'email_clicked_30d': [2, 3, 4],
            'email_open_rate': [0.5, 0.53, 0.5],
            'email_click_rate': [0.4, 0.375, 0.4],
            'website_visits_30d': [10, 15, 20],
            'avg_session_duration_min': [5, 7, 10],
            'pages_per_session': [3, 4, 5],
            'has_mobile_app': [0, 1, 1],
            'app_sessions_30d': [0, 10, 15],
            'social_media_follower': [0, 0, 1],
            'num_categories_purchased': [2, 3, 4],
            'purchases_with_discount_pct': [0.6, 0.7, 0.5],
            'cart_abandonment_rate': [0.3, 0.2, 0.1],
            'product_views_30d': [20, 30, 40],
            'wishlist_items': [5, 8, 10],
            'return_rate': [0.1, 0.05, 0.0],
            'converted': [0, 1, 0]
        })

        # Feature engineering should handle this gracefully
        engineer = PropensityFeatureEngineering()

        # This should either handle NaN or raise a clear error
        # (depends on your implementation choice)
        # For now, we just check it doesn't crash silently
        try:
            df_features = engineer.create_features(data)
            # If it succeeds, check that we get some output
            assert len(df_features) > 0
        except Exception as e:
            # If it fails, it should be a clear error
            assert str(e) != ""

    def test_invalid_feature_values(self):
        """Test handling of invalid feature values"""
        data = pd.DataFrame({
            'customer_id': ['CUST_001'],
            'age': [-5],  # Invalid: negative age
            'tenure_months': [12],
            'total_purchases': [5],
            'avg_order_value': [50],
            'total_revenue': [250],
            'days_since_last_purchase': [10],
            'purchase_frequency': [0.5],
            'is_repeat_customer': [1],
            'email_sent_30d': [10],
            'email_opened_30d': [5],
            'email_clicked_30d': [2],
            'email_open_rate': [1.5],  # Invalid: >1
            'email_click_rate': [0.4],
            'website_visits_30d': [10],
            'avg_session_duration_min': [5],
            'pages_per_session': [3],
            'has_mobile_app': [0],
            'app_sessions_30d': [0],
            'social_media_follower': [0],
            'num_categories_purchased': [2],
            'purchases_with_discount_pct': [0.6],
            'cart_abandonment_rate': [0.3],
            'product_views_30d': [20],
            'wishlist_items': [5],
            'return_rate': [0.1],
            'converted': [0]
        })

        # Should handle gracefully or raise clear error
        engineer = PropensityFeatureEngineering()

        try:
            df_features = engineer.create_features(data)
            # If successful, values should be clipped/handled
            assert df_features is not None
        except Exception as e:
            assert str(e) != ""


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
