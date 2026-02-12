"""
Model Explainability

SHAP values, feature importance, and prediction explanations
for understanding model decisions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import joblib
import json
from pathlib import Path


class PropensityExplainer:
    """Explain propensity model predictions"""

    def __init__(self, model, model_type, feature_names, X_sample):
        """
        Args:
            model: Trained model
            model_type: 'lightgbm', 'logistic_regression', or 'neural_network'
            feature_names: List of feature names
            X_sample: Sample of training data for SHAP baseline
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.X_sample = X_sample

        # Initialize SHAP explainer
        self._init_shap_explainer()

    def _init_shap_explainer(self):
        """Initialize appropriate SHAP explainer"""
        print(f"Initializing SHAP explainer for {self.model_type}...")

        if self.model_type == 'lightgbm':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'logistic_regression':
            self.explainer = shap.LinearExplainer(
                self.model,
                self.X_sample,
                feature_names=self.feature_names
            )
        else:  # neural_network
            # Use KernelExplainer for neural networks (slower but works)
            self.explainer = shap.KernelExplainer(
                self.model.predict if hasattr(self.model, 'predict') else lambda x: self.model(x).detach().numpy(),
                shap.sample(self.X_sample, 100)
            )

        print("✓ SHAP explainer ready")

    def explain_prediction(self, X, customer_id=None):
        """
        Explain a single prediction

        Args:
            X: Feature vector (1D array)
            customer_id: Optional customer identifier

        Returns:
            Dictionary with explanation details
        """
        # Ensure 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Get prediction
        if self.model_type == 'lightgbm':
            prediction = self.model.predict(X)[0]
        elif self.model_type == 'neural_network':
            import torch
            with torch.no_grad():
                prediction = self.model(torch.FloatTensor(X)).item()
        else:
            prediction = self.model.predict_proba(X)[0, 1]

        # Get SHAP values
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]

        # Top contributing features
        feature_contributions = pd.DataFrame({
            'feature': self.feature_names[:len(shap_values)],
            'value': X[0][:len(shap_values)],
            'shap_value': shap_values,
            'abs_shap': np.abs(shap_values)
        }).sort_values('abs_shap', ascending=False)

        explanation = {
            'customer_id': customer_id,
            'propensity_score': float(prediction),
            'top_positive_factors': feature_contributions[
                feature_contributions['shap_value'] > 0
            ].head(5)[['feature', 'value', 'shap_value']].to_dict('records'),
            'top_negative_factors': feature_contributions[
                feature_contributions['shap_value'] < 0
            ].head(5)[['feature', 'value', 'shap_value']].to_dict('records'),
            'all_contributions': feature_contributions.to_dict('records')
        }

        return explanation

    def global_feature_importance(self, X, top_n=20):
        """
        Calculate global feature importance across dataset

        Args:
            X: Feature matrix
            top_n: Number of top features to return
        """
        print("Calculating global feature importance...")

        # Get SHAP values for sample
        sample_size = min(1000, len(X))
        X_sample = X[:sample_size]

        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Calculate mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(mean_abs_shap)],
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        print(f"\nTop {top_n} Most Important Features:")
        print(importance_df.head(top_n).to_string(index=False))

        return importance_df

    def plot_feature_importance(self, X, save_path=None, top_n=20):
        """Plot global feature importance"""
        importance_df = self.global_feature_importance(X, top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=importance_df.head(top_n),
            y='feature',
            x='importance',
            palette='viridis'
        )
        plt.xlabel('Mean |SHAP Value|')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance - {self.model_type.upper()}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")

        return plt

    def plot_shap_summary(self, X, save_path=None, max_display=20):
        """
        Create SHAP summary plot showing feature impacts

        Args:
            X: Feature matrix
            save_path: Path to save plot
            max_display: Number of features to display
        """
        print("Creating SHAP summary plot...")

        # Sample for visualization
        sample_size = min(1000, len(X))
        X_sample = X[:sample_size]

        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        else:
            plt.show()

        return plt

    def plot_waterfall(self, X, idx=0, save_path=None):
        """
        Create waterfall plot for a single prediction

        Shows how each feature contributes to moving from base value to prediction
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        shap_values = self.explainer.shap_values(X[idx:idx+1])

        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0]

        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            data=X[idx],
            feature_names=self.feature_names
        )

        shap.waterfall_plot(explanation, max_display=15, show=False)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved waterfall plot to {save_path}")
        else:
            plt.show()

        return plt

    def plot_dependence(self, X, feature_name, save_path=None):
        """
        Create dependence plot showing how a feature affects predictions

        Args:
            X: Feature matrix
            feature_name: Name of feature to analyze
            save_path: Path to save plot
        """
        if feature_name not in self.feature_names:
            print(f"Feature '{feature_name}' not found")
            return

        feature_idx = self.feature_names.index(feature_name)

        # Sample for visualization
        sample_size = min(1000, len(X))
        X_sample = X[:sample_size]

        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_sample,
            feature_names=self.feature_names,
            show=False
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved dependence plot to {save_path}")
        else:
            plt.show()

        return plt

    def analyze_segment(self, X, y_pred, segment_name, segment_mask):
        """
        Analyze feature importance for a specific customer segment

        Args:
            X: Feature matrix
            y_pred: Predictions
            segment_name: Name of segment
            segment_mask: Boolean mask for segment
        """
        print(f"\nAnalyzing segment: {segment_name}")
        print(f"Segment size: {segment_mask.sum():,} ({segment_mask.mean():.1%})")

        X_segment = X[segment_mask]

        if len(X_segment) == 0:
            print("Segment is empty")
            return

        # Get SHAP values for segment
        sample_size = min(500, len(X_segment))
        shap_values = self.explainer.shap_values(X_segment[:sample_size])

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Calculate mean importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(mean_abs_shap)],
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        print("\nTop 10 features for this segment:")
        print(importance_df.head(10).to_string(index=False))

        return importance_df


def explain_model(model, model_type, X_train, X_test, feature_names, output_dir='plots'):
    """
    Generate comprehensive model explanations

    Args:
        model: Trained model
        model_type: Type of model
        X_train: Training features (for baseline)
        X_test: Test features
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    print("=" * 60)
    print("MODEL EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Initialize explainer
    explainer = PropensityExplainer(model, model_type, feature_names, X_train)

    # 1. Global feature importance
    print("\n1. Global Feature Importance")
    explainer.plot_feature_importance(
        X_test,
        save_path=f'{output_dir}/{model_type}_feature_importance.png'
    )
    plt.close()

    # 2. SHAP summary plot
    print("\n2. SHAP Summary Plot")
    explainer.plot_shap_summary(
        X_test,
        save_path=f'{output_dir}/{model_type}_shap_summary.png'
    )
    plt.close()

    # 3. Example waterfall plots
    print("\n3. Individual Prediction Explanations")
    for i in [0, 10, 50]:
        if i < len(X_test):
            explainer.plot_waterfall(
                X_test,
                idx=i,
                save_path=f'{output_dir}/{model_type}_waterfall_example_{i}.png'
            )
            plt.close()

    # 4. Top feature dependence plots
    print("\n4. Feature Dependence Plots")
    importance_df = explainer.global_feature_importance(X_test, top_n=5)

    for feature in importance_df.head(3)['feature']:
        explainer.plot_dependence(
            X_test,
            feature,
            save_path=f'{output_dir}/{model_type}_dependence_{feature}.png'
        )
        plt.close()

    print("\n" + "=" * 60)
    print("✓ Explainability analysis complete!")
    print(f"✓ Plots saved to {output_dir}/")
    print("=" * 60)

    return explainer


if __name__ == "__main__":
    # Test explainability
    print("Testing model explainability...")

    # Load model and data
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.models.propensity_model import train_propensity_models

    # Train model
    trainer, results = train_propensity_models()

    # Explain models
    for model_name in ['lightgbm', 'logistic_regression']:
        print(f"\n\nExplaining {model_name}...")
        explain_model(
            model=trainer.models[model_name],
            model_type=model_name,
            X_train=results['X_train'],
            X_test=results['X_test'],
            feature_names=trainer.feature_names,
            output_dir=f'plots/{model_name}'
        )
