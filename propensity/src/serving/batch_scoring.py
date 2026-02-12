"""
Batch Scoring Pipeline

Efficiently score large batches of customers for propensity.
Designed for daily/weekly scoring jobs.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import joblib
from datetime import datetime
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data_pipeline.feature_engineering import PropensityFeatureEngineering


class BatchScorer:
    """Score customers in batches for propensity"""

    def __init__(self, model_path, model_type='lightgbm', batch_size=10000):
        """
        Args:
            model_path: Path to saved model
            model_type: Type of model ('lightgbm', 'logistic_regression', 'neural_network')
            batch_size: Number of customers per batch
        """
        self.model_path = model_path
        self.model_type = model_type
        self.batch_size = batch_size
        self.model = None
        self.feature_engineer = PropensityFeatureEngineering()
        self.feature_names = None

        self._load_model()

    def _load_model(self):
        """Load the trained model"""
        print(f"Loading {self.model_type} model from {self.model_path}...")

        if self.model_type == 'lightgbm':
            self.model = lgb.Booster(model_file=self.model_path)
        elif self.model_type == 'neural_network':
            # Would need model architecture definition
            # For now, skip neural network in batch scoring
            raise NotImplementedError("Neural network batch scoring not implemented")
        else:
            self.model = joblib.load(self.model_path)

        # Load metadata if available
        metadata_path = Path(self.model_path).parent / 'model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])

        print("✓ Model loaded successfully")

    def score_batch(self, df):
        """
        Score a batch of customers

        Args:
            df: DataFrame with raw customer features

        Returns:
            DataFrame with propensity scores
        """
        # Engineer features
        df_features = self.feature_engineer.create_features(df)

        # Select features
        if self.feature_names:
            feature_cols = [col for col in self.feature_names if col in df_features.columns]
        else:
            feature_cols = df_features.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in feature_cols if col not in ['customer_id', 'converted']]

        X = df_features[feature_cols].values

        # Predict
        if self.model_type == 'lightgbm':
            propensity_scores = self.model.predict(X)
        else:
            propensity_scores = self.model.predict_proba(X)[:, 1]

        # Create result DataFrame
        results = pd.DataFrame({
            'customer_id': df['customer_id'],
            'propensity_score': propensity_scores,
            'risk_category': pd.cut(
                propensity_scores,
                bins=[0, 0.4, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            ),
            'score_date': datetime.now().strftime('%Y-%m-%d'),
            'model_type': self.model_type
        })

        return results

    def score_file(self, input_path, output_path, chunksize=None):
        """
        Score customers from a parquet/CSV file

        Args:
            input_path: Path to input data
            output_path: Path to save results
            chunksize: Process in chunks (for very large files)
        """
        print("=" * 60)
        print("BATCH SCORING")
        print("=" * 60)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Model:  {self.model_type}")
        print("=" * 60)

        # Determine file type
        if input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)

        print(f"Loaded {len(df):,} customers")

        # Score in batches
        results_list = []

        for i in tqdm(range(0, len(df), self.batch_size), desc="Scoring batches"):
            batch = df.iloc[i:i + self.batch_size]
            batch_results = self.score_batch(batch)
            results_list.append(batch_results)

        # Combine results
        all_results = pd.concat(results_list, ignore_index=True)

        # Save results
        if output_path.endswith('.parquet'):
            all_results.to_parquet(output_path, index=False)
        else:
            all_results.to_csv(output_path, index=False)

        print(f"\n✓ Scored {len(all_results):,} customers")
        print(f"✓ Results saved to {output_path}")

        # Summary statistics
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, results):
        """Print scoring summary statistics"""
        print("\n" + "=" * 60)
        print("SCORING SUMMARY")
        print("=" * 60)

        print("\nPropensity Score Distribution:")
        print(results['propensity_score'].describe())

        print("\nRisk Category Breakdown:")
        category_counts = results['risk_category'].value_counts()
        for category, count in category_counts.items():
            pct = count / len(results) * 100
            print(f"  {category:10s}: {count:7,} ({pct:5.1f}%)")

        print("\nTop 10 Highest Propensity Customers:")
        top_customers = results.nlargest(10, 'propensity_score')[
            ['customer_id', 'propensity_score', 'risk_category']
        ]
        print(top_customers.to_string(index=False))


def score_customers_daily(
    customer_data_path='data/raw/customer_data.parquet',
    model_path='models/lightgbm_latest.txt',
    output_dir='data/scores',
    model_type='lightgbm'
):
    """
    Daily batch scoring job

    Args:
        customer_data_path: Path to customer data
        model_path: Path to trained model
        output_dir: Directory to save scores
        model_type: Type of model
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Output filename with date
    today = datetime.now().strftime('%Y%m%d')
    output_path = f"{output_dir}/propensity_scores_{today}.parquet"

    # Initialize scorer
    scorer = BatchScorer(model_path, model_type=model_type, batch_size=10000)

    # Score customers
    results = scorer.score_file(customer_data_path, output_path)

    # Generate business report
    generate_business_report(results, output_dir, today)

    return results


def generate_business_report(results, output_dir, date):
    """
    Generate business-friendly report from scores

    Args:
        results: Scored customer DataFrame
        output_dir: Directory to save report
        date: Date string for filename
    """
    report_path = f"{output_dir}/business_report_{date}.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PROPENSITY SCORING BUSINESS REPORT\n")
        f.write(f"Date: {date}\n")
        f.write("=" * 70 + "\n\n")

        # Summary metrics
        total_customers = len(results)
        high_propensity = (results['propensity_score'] >= 0.7).sum()
        medium_propensity = ((results['propensity_score'] >= 0.4) & (results['propensity_score'] < 0.7)).sum()
        low_propensity = (results['propensity_score'] < 0.4).sum()

        f.write("CUSTOMER SEGMENTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Customers:          {total_customers:10,}\n\n")
        f.write(f"High Propensity (≥70%):   {high_propensity:10,} ({high_propensity/total_customers*100:5.1f}%)\n")
        f.write(f"Medium Propensity (40-70%): {medium_propensity:10,} ({medium_propensity/total_customers*100:5.1f}%)\n")
        f.write(f"Low Propensity (<40%):    {low_propensity:10,} ({low_propensity/total_customers*100:5.1f}%)\n\n")

        # Action recommendations
        f.write("RECOMMENDED ACTIONS\n")
        f.write("-" * 70 + "\n")
        f.write(f"1. HIGH PROPENSITY CUSTOMERS ({high_propensity:,}):\n")
        f.write("   → Immediate outreach with premium offers\n")
        f.write("   → Personalized email campaigns\n")
        f.write("   → Sales team prioritization\n")
        f.write("   → Expected conversion rate: 70-90%\n\n")

        f.write(f"2. MEDIUM PROPENSITY CUSTOMERS ({medium_propensity:,}):\n")
        f.write("   → Targeted nurture campaigns\n")
        f.write("   → Limited-time promotions\n")
        f.write("   → Product education content\n")
        f.write("   → Expected conversion rate: 40-60%\n\n")

        f.write(f"3. LOW PROPENSITY CUSTOMERS ({low_propensity:,}):\n")
        f.write("   → Generic awareness campaigns\n")
        f.write("   → Minimal marketing spend\n")
        f.write("   → Focus on engagement building\n\n")

        # Top customers
        f.write("TOP 20 HIGH-PROPENSITY CUSTOMERS\n")
        f.write("-" * 70 + "\n")
        top_customers = results.nlargest(20, 'propensity_score')
        for idx, row in top_customers.iterrows():
            f.write(f"{row['customer_id']:15s}  Score: {row['propensity_score']:.1%}  Category: {row['risk_category']}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Report generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 70 + "\n")

    print(f"\n✓ Business report saved to {report_path}")


def main():
    """Command-line interface for batch scoring"""
    parser = argparse.ArgumentParser(description='Batch score customers for propensity')

    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/customer_data.parquet',
        help='Path to customer data file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/scores',
        help='Output directory for scores'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='models/lightgbm_latest.txt',
        help='Path to trained model'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        default='lightgbm',
        choices=['lightgbm', 'logistic_regression'],
        help='Type of model'
    )

    args = parser.parse_args()

    # Run scoring
    score_customers_daily(
        customer_data_path=args.input,
        model_path=args.model,
        output_dir=args.output,
        model_type=args.model_type
    )


if __name__ == "__main__":
    main()
