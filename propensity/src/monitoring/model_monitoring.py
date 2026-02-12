"""
Model Monitoring

Monitors model performance, data drift, and prediction quality in production.
Includes:
- Data drift detection (PSI)
- Model performance tracking
- Prediction distribution monitoring
- Alerting on degradation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from pathlib import Path
import warnings


class DataDriftDetector:
    """Detect data drift using Population Stability Index (PSI)"""

    def __init__(self, reference_data: pd.DataFrame, n_bins: int = 10):
        """
        Args:
            reference_data: Training/baseline data
            n_bins: Number of bins for PSI calculation
        """
        self.reference_data = reference_data
        self.n_bins = n_bins
        self.reference_distributions = {}

        self._calculate_reference_distributions()

    def _calculate_reference_distributions(self):
        """Calculate reference distributions for all features"""
        for col in self.reference_data.select_dtypes(include=[np.number]).columns:
            if col not in ['customer_id', 'converted']:
                # Calculate bins
                _, bins = pd.cut(
                    self.reference_data[col],
                    bins=self.n_bins,
                    retbins=True,
                    duplicates='drop'
                )

                # Calculate distribution
                ref_dist = pd.cut(
                    self.reference_data[col],
                    bins=bins,
                    include_lowest=True
                ).value_counts(normalize=True).sort_index()

                self.reference_distributions[col] = {
                    'bins': bins,
                    'distribution': ref_dist
                }

    def calculate_psi(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate PSI for all features

        PSI = Σ (current% - reference%) * ln(current% / reference%)

        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.2: Moderate change
        - PSI ≥ 0.2: Significant change (investigate!)

        Args:
            current_data: New production data

        Returns:
            Dictionary of feature -> PSI value
        """
        psi_values = {}

        for col, ref_info in self.reference_distributions.items():
            if col not in current_data.columns:
                continue

            # Calculate current distribution
            current_dist = pd.cut(
                current_data[col],
                bins=ref_info['bins'],
                include_lowest=True
            ).value_counts(normalize=True).sort_index()

            # Align distributions
            ref_dist = ref_info['distribution']

            # Calculate PSI
            psi = 0
            for category in ref_dist.index:
                ref_pct = ref_dist.get(category, 0.0001)  # Avoid division by zero
                curr_pct = current_dist.get(category, 0.0001)

                psi += (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)

            psi_values[col] = psi

        return psi_values

    def detect_drift(self, current_data: pd.DataFrame, threshold: float = 0.2) -> Dict:
        """
        Detect features with significant drift

        Args:
            current_data: New production data
            threshold: PSI threshold for alerting

        Returns:
            Dictionary with drift analysis
        """
        psi_values = self.calculate_psi(current_data)

        # Classify drift severity
        drifted_features = {
            'significant': [],  # PSI >= 0.2
            'moderate': [],     # 0.1 <= PSI < 0.2
            'stable': []        # PSI < 0.1
        }

        for feature, psi in psi_values.items():
            if psi >= 0.2:
                drifted_features['significant'].append((feature, psi))
            elif psi >= 0.1:
                drifted_features['moderate'].append((feature, psi))
            else:
                drifted_features['stable'].append((feature, psi))

        # Sort by PSI value
        for category in drifted_features:
            drifted_features[category] = sorted(
                drifted_features[category],
                key=lambda x: x[1],
                reverse=True
            )

        return {
            'psi_values': psi_values,
            'drifted_features': drifted_features,
            'alert': len(drifted_features['significant']) > 0,
            'timestamp': datetime.now().isoformat()
        }


class ModelPerformanceMonitor:
    """Monitor model performance over time"""

    def __init__(self, baseline_metrics: Dict):
        """
        Args:
            baseline_metrics: Performance metrics from initial evaluation
        """
        self.baseline_metrics = baseline_metrics
        self.performance_history = []

    def log_performance(self, y_true: np.ndarray, y_pred: np.ndarray, timestamp: str = None):
        """
        Log model performance for a batch of predictions

        Args:
            y_true: Actual labels
            y_pred: Predicted probabilities
            timestamp: Time of predictions
        """
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Calculate metrics
        metrics = {
            'timestamp': timestamp,
            'n_samples': len(y_true),
            'positive_rate': y_true.mean(),
            'auc': roc_auc_score(y_true, y_pred),
            'precision': precision_score(y_true, (y_pred >= 0.5).astype(int)),
            'recall': recall_score(y_true, (y_pred >= 0.5).astype(int)),
            'f1': f1_score(y_true, (y_pred >= 0.5).astype(int))
        }

        self.performance_history.append(metrics)

        return metrics

    def check_degradation(self, current_metrics: Dict, threshold: float = 0.05) -> Dict:
        """
        Check if model performance has degraded

        Args:
            current_metrics: Recent performance metrics
            threshold: Acceptable degradation (e.g., 0.05 = 5% drop)

        Returns:
            Degradation analysis
        """
        degradation = {}

        for metric, baseline_value in self.baseline_metrics.items():
            if metric in current_metrics and isinstance(baseline_value, (int, float)):
                current_value = current_metrics[metric]
                change = (current_value - baseline_value) / baseline_value

                degradation[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'change_pct': change * 100,
                    'degraded': change < -threshold
                }

        # Overall alert
        alert = any(m['degraded'] for m in degradation.values())

        return {
            'degradation': degradation,
            'alert': alert,
            'timestamp': datetime.now().isoformat()
        }

    def get_performance_summary(self, days: int = 7) -> pd.DataFrame:
        """
        Get performance summary for last N days

        Args:
            days: Number of days to include

        Returns:
            DataFrame with performance metrics
        """
        if not self.performance_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.performance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter last N days
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] >= cutoff]

        return df


class PredictionDistributionMonitor:
    """Monitor prediction score distributions"""

    def __init__(self, reference_predictions: np.ndarray):
        """
        Args:
            reference_predictions: Predictions from training/validation set
        """
        self.reference_predictions = reference_predictions
        self.reference_stats = {
            'mean': reference_predictions.mean(),
            'std': reference_predictions.std(),
            'percentiles': np.percentile(reference_predictions, [10, 25, 50, 75, 90])
        }

    def check_distribution(self, current_predictions: np.ndarray) -> Dict:
        """
        Check if prediction distribution has shifted

        Args:
            current_predictions: Recent predictions

        Returns:
            Distribution analysis
        """
        current_stats = {
            'mean': current_predictions.mean(),
            'std': current_predictions.std(),
            'percentiles': np.percentile(current_predictions, [10, 25, 50, 75, 90])
        }

        # Calculate shifts
        mean_shift = abs(current_stats['mean'] - self.reference_stats['mean'])
        std_shift = abs(current_stats['std'] - self.reference_stats['std'])

        # Alert if mean shifts by more than 0.1 or std shifts by more than 0.05
        alert = mean_shift > 0.1 or std_shift > 0.05

        return {
            'reference_stats': self.reference_stats,
            'current_stats': current_stats,
            'mean_shift': mean_shift,
            'std_shift': std_shift,
            'alert': alert,
            'timestamp': datetime.now().isoformat()
        }


class PropensityMonitor:
    """Complete monitoring system for propensity models"""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        baseline_metrics: Dict,
        reference_predictions: np.ndarray
    ):
        """
        Args:
            reference_data: Training/baseline data
            baseline_metrics: Initial model performance
            reference_predictions: Baseline predictions
        """
        self.drift_detector = DataDriftDetector(reference_data)
        self.performance_monitor = ModelPerformanceMonitor(baseline_metrics)
        self.distribution_monitor = PredictionDistributionMonitor(reference_predictions)

        self.monitoring_history = []

    def monitor(
        self,
        current_data: pd.DataFrame,
        predictions: np.ndarray,
        actual_labels: np.ndarray = None
    ) -> Dict:
        """
        Run complete monitoring check

        Args:
            current_data: New production data
            predictions: Model predictions
            actual_labels: Actual outcomes (if available)

        Returns:
            Complete monitoring report
        """
        print("Running monitoring checks...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(current_data)
        }

        # 1. Data drift
        print("  - Checking data drift...")
        drift_report = self.drift_detector.detect_drift(current_data)
        report['data_drift'] = drift_report

        # 2. Prediction distribution
        print("  - Checking prediction distribution...")
        dist_report = self.distribution_monitor.check_distribution(predictions)
        report['prediction_distribution'] = dist_report

        # 3. Model performance (if labels available)
        if actual_labels is not None:
            print("  - Checking model performance...")
            perf_metrics = self.performance_monitor.log_performance(actual_labels, predictions)
            degradation_report = self.performance_monitor.check_degradation(perf_metrics)
            report['performance'] = degradation_report

        # Overall alert
        report['alert'] = (
            drift_report.get('alert', False) or
            dist_report.get('alert', False) or
            (report.get('performance', {}).get('alert', False) if actual_labels is not None else False)
        )

        # Store in history
        self.monitoring_history.append(report)

        # Print summary
        self._print_report(report)

        return report

    def _print_report(self, report: Dict):
        """Print monitoring report"""
        print("\n" + "=" * 60)
        print("MONITORING REPORT")
        print("=" * 60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Samples:   {report['n_samples']:,}")

        # Data drift
        drift = report['data_drift']
        print(f"\nData Drift:")
        print(f"  Significant: {len(drift['drifted_features']['significant'])} features")
        print(f"  Moderate:    {len(drift['drifted_features']['moderate'])} features")

        if drift['drifted_features']['significant']:
            print(f"\n  ⚠️  Significant Drift Detected:")
            for feature, psi in drift['drifted_features']['significant'][:5]:
                print(f"     • {feature}: PSI = {psi:.3f}")

        # Prediction distribution
        dist = report['prediction_distribution']
        print(f"\nPrediction Distribution:")
        print(f"  Mean shift: {dist['mean_shift']:.4f}")
        print(f"  Std shift:  {dist['std_shift']:.4f}")

        if dist['alert']:
            print(f"  ⚠️  Distribution shift detected!")

        # Performance
        if 'performance' in report:
            perf = report['performance']
            print(f"\nModel Performance:")
            for metric, values in perf['degradation'].items():
                if isinstance(values, dict):
                    change = values['change_pct']
                    symbol = "⚠️ " if values['degraded'] else "✓ "
                    print(f"  {symbol} {metric}: {change:+.1f}%")

        # Overall status
        print(f"\n{'⚠️  ALERT' if report['alert'] else '✓ OK'}")
        print("=" * 60)

    def save_report(self, report: Dict, output_path: str):
        """Save monitoring report to file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    # Test monitoring
    print("Testing monitoring system...")

    # Generate sample data
    from data.sample_generator import CustomerDataGenerator

    generator = CustomerDataGenerator(n_customers=10000)
    baseline_data = generator.generate()

    # Simulate predictions
    baseline_predictions = np.random.beta(2, 5, len(baseline_data))

    # Initialize monitor
    baseline_metrics = {
        'auc': 0.85,
        'precision': 0.75,
        'recall': 0.70,
        'f1': 0.72
    }

    monitor = PropensityMonitor(
        reference_data=baseline_data,
        baseline_metrics=baseline_metrics,
        reference_predictions=baseline_predictions
    )

    # Simulate new data with some drift
    current_data = generator.generate()
    current_data['age'] = current_data['age'] + 5  # Age drift
    current_predictions = np.random.beta(2.5, 5, len(current_data))  # Distribution shift

    # Run monitoring
    report = monitor.monitor(current_data, current_predictions)

    print("\n✓ Monitoring test complete")
