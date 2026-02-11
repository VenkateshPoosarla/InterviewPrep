"""
Monitoring and Observability for Recommendation Systems

Staff Interview Topics:
- Model performance monitoring
- Data drift detection
- Online vs offline metric correlation
- Alerting and incident response
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring"""
    # Drift detection
    drift_detection_window_days: int = 7
    drift_threshold: float = 0.1

    # Performance monitoring
    latency_p99_threshold_ms: float = 100.0
    min_daily_requests: int = 10000

    # Model performance
    min_ctr_threshold: float = 0.02
    min_conversion_threshold: float = 0.01


class DataDriftDetector:
    """
    Detect distribution shifts in features

    Interview Topic: Why does drift matter?
    - Model trained on old distribution
    - Performance degradation over time
    - Need to trigger retraining

    Methods:
    - KL Divergence (continuous features)
    - Chi-square test (categorical features)
    - Population Stability Index (PSI)
    """

    @staticmethod
    def compute_psi(
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Population Stability Index

        PSI = sum((current% - baseline%) * ln(current% / baseline%))

        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 < PSI < 0.2: Moderate change
        - PSI > 0.2: Significant change (retrain needed)

        Interview Point: Industry standard for drift detection
        """
        # Create bins
        breakpoints = np.percentile(baseline, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)

        # Count samples in each bin
        baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
        current_counts = np.histogram(current, bins=breakpoints)[0]

        # Convert to percentages
        baseline_pct = baseline_counts / len(baseline)
        current_pct = current_counts / len(current)

        # Avoid division by zero
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)

        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return float(psi)

    @staticmethod
    def detect_categorical_drift(
        baseline: pd.Series,
        current: pd.Series,
        significance: float = 0.05
    ) -> Dict:
        """
        Chi-square test for categorical features

        Returns:
            {
                'drift_detected': bool,
                'p_value': float,
                'statistic': float
            }
        """
        # Get value counts
        baseline_counts = baseline.value_counts()
        current_counts = current.value_counts()

        # Align categories
        all_categories = set(baseline_counts.index) | set(current_counts.index)
        baseline_aligned = [baseline_counts.get(cat, 0) for cat in all_categories]
        current_aligned = [current_counts.get(cat, 0) for cat in all_categories]

        # Chi-square test
        statistic, p_value = stats.chisquare(current_aligned, baseline_aligned)

        return {
            'drift_detected': p_value < significance,
            'p_value': float(p_value),
            'statistic': float(statistic)
        }


class PerformanceMonitor:
    """
    Monitor model performance in production

    Interview Topic: Online vs Offline metrics
    - Offline: NDCG, AUC (computed on test set)
    - Online: CTR, conversion, engagement (real user behavior)
    - Correlation is not perfect!
    """

    @staticmethod
    def compute_online_metrics(
        recommendations_df: pd.DataFrame,
        interactions_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute online metrics from recommendation logs

        Args:
            recommendations_df: Columns [request_id, user_id, item_id, rank, timestamp]
            interactions_df: Columns [user_id, item_id, event_type, timestamp]

        Returns:
            Dictionary of online metrics
        """
        # Join recommendations with interactions
        merged = recommendations_df.merge(
            interactions_df,
            on=['user_id', 'item_id'],
            how='left',
            suffixes=('_rec', '_int')
        )

        # Filter interactions within time window (e.g., 1 hour after recommendation)
        merged['time_diff'] = (
            merged['timestamp_int'] - merged['timestamp_rec']
        ).dt.total_seconds()

        # Only count interactions within 1 hour
        merged = merged[
            (merged['time_diff'] >= 0) &
            (merged['time_diff'] <= 3600)
        ]

        # Compute metrics
        total_recs = len(recommendations_df)

        # CTR (clicks / impressions)
        clicks = merged[merged['event_type'] == 'click'].shape[0]
        ctr = clicks / total_recs if total_recs > 0 else 0

        # Conversion rate
        conversions = merged[merged['event_type'] == 'purchase'].shape[0]
        conversion_rate = conversions / total_recs if total_recs > 0 else 0

        # Average rank of clicked items (lower is better)
        clicked_ranks = merged[merged['event_type'] == 'click']['rank']
        avg_clicked_rank = clicked_ranks.mean() if len(clicked_ranks) > 0 else 0

        # Engagement rate (any interaction)
        engagements = merged[merged['event_type'].notna()].shape[0]
        engagement_rate = engagements / total_recs if total_recs > 0 else 0

        return {
            'ctr': ctr,
            'conversion_rate': conversion_rate,
            'engagement_rate': engagement_rate,
            'avg_clicked_rank': avg_clicked_rank,
            'total_recommendations': total_recs,
            'total_clicks': clicks,
            'total_conversions': conversions
        }

    @staticmethod
    def check_degradation(
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        threshold: float = 0.1
    ) -> List[str]:
        """
        Check if metrics have degraded significantly

        Returns:
            List of degraded metric names

        Interview Point: Statistical significance vs business significance
        """
        degraded_metrics = []

        for metric_name in ['ctr', 'conversion_rate', 'engagement_rate']:
            current = current_metrics.get(metric_name, 0)
            baseline = baseline_metrics.get(metric_name, 0)

            if baseline > 0:
                change_pct = (current - baseline) / baseline

                if change_pct < -threshold:  # Degraded by more than threshold
                    degraded_metrics.append(
                        f"{metric_name}: {change_pct*100:.1f}% decrease"
                    )

        return degraded_metrics


class ABTestMonitor:
    """
    A/B test monitoring and analysis

    Interview Topic: Statistical rigor in experimentation
    - Sample size calculation
    - Statistical significance
    - Multiple testing correction
    - Early stopping
    """

    @staticmethod
    def calculate_required_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        Calculate required sample size for A/B test

        Args:
            baseline_rate: Current conversion/CTR rate
            minimum_detectable_effect: Smallest effect we want to detect (e.g., 0.05 for 5%)
            alpha: Significance level
            power: Statistical power

        Interview Point: Why you can't just run tests for a day
        """
        from statsmodels.stats.power import zt_ind_solve_power

        # Effect size (Cohen's h for proportions)
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        # Calculate sample size per variant
        n = zt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative='two-sided'
        )

        return int(np.ceil(n))

    @staticmethod
    def analyze_ab_test(
        control_conversions: int,
        control_samples: int,
        treatment_conversions: int,
        treatment_samples: int
    ) -> Dict:
        """
        Analyze A/B test results

        Returns:
            {
                'control_rate': float,
                'treatment_rate': float,
                'relative_lift': float,
                'p_value': float,
                'is_significant': bool,
                'confidence_interval': tuple
            }

        Interview Topic: When to ship experiment
        """
        control_rate = control_conversions / control_samples
        treatment_rate = treatment_conversions / treatment_samples

        # Relative lift
        relative_lift = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0

        # Z-test for proportions
        pooled_rate = (control_conversions + treatment_conversions) / (control_samples + treatment_samples)

        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_samples + 1/treatment_samples))
        z_score = (treatment_rate - control_rate) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Confidence interval (95%)
        se_diff = np.sqrt(
            control_rate * (1 - control_rate) / control_samples +
            treatment_rate * (1 - treatment_rate) / treatment_samples
        )
        ci_lower = (treatment_rate - control_rate) - 1.96 * se_diff
        ci_upper = (treatment_rate - control_rate) + 1.96 * se_diff

        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'relative_lift': relative_lift,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'confidence_interval': (ci_lower, ci_upper),
            'recommendation': 'Ship' if p_value < 0.05 and relative_lift > 0 else 'Do not ship'
        }


# Example usage
if __name__ == "__main__":
    # Example: Detect drift
    baseline_data = np.random.normal(0, 1, 10000)
    current_data = np.random.normal(0.2, 1.1, 10000)  # Shifted distribution

    psi = DataDriftDetector.compute_psi(baseline_data, current_data)
    print(f"PSI: {psi:.4f}")

    if psi > 0.2:
        print("⚠️  Significant drift detected! Retrain model.")

    # Example: A/B test sample size
    required_n = ABTestMonitor.calculate_required_sample_size(
        baseline_rate=0.03,
        minimum_detectable_effect=0.05
    )
    print(f"Required sample size per variant: {required_n:,}")
