"""
End-to-End Propensity Modeling Demo

Demonstrates the complete ML pipeline:
1. Generate synthetic customer data
2. Engineer features
3. Train multiple models
4. Evaluate and compare
5. Generate explanations
6. Make predictions
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.sample_generator import CustomerDataGenerator, generate_time_series_data
from data_pipeline.feature_engineering import create_propensity_features
from models.propensity_model import PropensityModelTrainer
from models.model_explainability import explain_model


def run_demo():
    """Run complete propensity modeling pipeline"""
    print("=" * 80)
    print("PROPENSITY MODELING - END-TO-END DEMO")
    print("=" * 80)

    # ===========================
    # STEP 1: Generate Data
    # ===========================
    print("\n" + "=" * 80)
    print("STEP 1: GENERATE SYNTHETIC CUSTOMER DATA")
    print("=" * 80)

    generator = CustomerDataGenerator(n_customers=50000)
    customer_data = generator.generate()

    # Save raw data
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    customer_data.to_parquet('data/raw/customer_data.parquet', index=False)

    print("\nData Preview:")
    print(customer_data.head())
    print(f"\nShape: {customer_data.shape}")
    print(f"Conversion Rate: {customer_data['converted'].mean():.2%}")

    # ===========================
    # STEP 2: Feature Engineering
    # ===========================
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 80)

    df_features, feature_names = create_propensity_features(customer_data, scale=True)

    # Save processed features
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df_features.to_parquet('data/processed/customer_features.parquet', index=False)

    print(f"\n✓ Created {len(feature_names)} features")
    print("\nSample Features:")
    print(df_features[feature_names[:10]].head())

    # ===========================
    # STEP 3: Train Models
    # ===========================
    print("\n" + "=" * 80)
    print("STEP 3: TRAIN PROPENSITY MODELS")
    print("=" * 80)

    trainer = PropensityModelTrainer()

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df_features)

    # Train all models
    print("\n--- Training Logistic Regression ---")
    trainer.train_logistic_regression(X_train, y_train, X_val, y_val)

    print("\n--- Training LightGBM ---")
    trainer.train_lightgbm(X_train, y_train, X_val, y_val)

    print("\n--- Training Neural Network ---")
    trainer.train_neural_network(X_train, y_train, X_val, y_val, epochs=30)

    # ===========================
    # STEP 4: Evaluate Models
    # ===========================
    print("\n" + "=" * 80)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 80)

    test_results = trainer.evaluate_on_test(X_test, y_test)

    # Compare models
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    comparison = pd.DataFrame({
        model_name: {
            'Validation AUC': trainer.metrics[model_name]['val_auc'],
            'Test AUC': test_results[model_name]['auc']
        }
        for model_name in trainer.models.keys()
    }).T

    print(comparison)

    # Find best model
    best_model = comparison['Test AUC'].idxmax()
    print(f"\n🏆 Best Model: {best_model.upper()}")
    print(f"   Test AUC: {comparison.loc[best_model, 'Test AUC']:.4f}")

    # ===========================
    # STEP 5: Model Explainability
    # ===========================
    print("\n" + "=" * 80)
    print("STEP 5: MODEL EXPLAINABILITY")
    print("=" * 80)

    # Create plots directory
    Path('plots').mkdir(exist_ok=True)

    # Explain best model
    print(f"\nGenerating explanations for {best_model}...")

    try:
        explainer = explain_model(
            model=trainer.models[best_model],
            model_type=best_model,
            X_train=X_train,
            X_test=X_test,
            feature_names=trainer.feature_names,
            output_dir=f'plots/{best_model}'
        )

        # Show example prediction explanation
        print("\n" + "=" * 80)
        print("EXAMPLE PREDICTION EXPLANATION")
        print("=" * 80)

        sample_idx = 0
        explanation = explainer.explain_prediction(X_test[sample_idx], customer_id=f"CUST_{sample_idx}")

        print(f"\nCustomer ID: {explanation['customer_id']}")
        print(f"Propensity Score: {explanation['propensity_score']:.2%}")

        print("\nTop Positive Factors (Increase Conversion Probability):")
        for factor in explanation['top_positive_factors'][:5]:
            print(f"  • {factor['feature']}: {factor['shap_value']:.4f}")

        print("\nTop Negative Factors (Decrease Conversion Probability):")
        for factor in explanation['top_negative_factors'][:5]:
            print(f"  • {factor['feature']}: {factor['shap_value']:.4f}")

    except Exception as e:
        print(f"Warning: Could not generate explanations: {e}")
        print("Continuing with demo...")

    # ===========================
    # STEP 6: Save Models
    # ===========================
    print("\n" + "=" * 80)
    print("STEP 6: SAVE MODELS")
    print("=" * 80)

    trainer.save_models()

    # ===========================
    # STEP 7: Make Predictions
    # ===========================
    print("\n" + "=" * 80)
    print("STEP 7: PREDICTION EXAMPLES")
    print("=" * 80)

    # Sample predictions on test set
    print("\nSample Predictions on Test Set:")

    # Get predictions from best model
    if best_model == 'lightgbm':
        predictions = trainer.models[best_model].predict(X_test[:10])
    elif best_model == 'neural_network':
        import torch
        with torch.no_grad():
            predictions = trainer.models[best_model](torch.FloatTensor(X_test[:10])).numpy()
    else:
        predictions = trainer.models[best_model].predict_proba(X_test[:10])[:, 1]

    prediction_df = pd.DataFrame({
        'Customer': [f"CUST_{i}" for i in range(10)],
        'Propensity Score': predictions,
        'Risk Category': ['High' if p >= 0.7 else 'Medium' if p >= 0.4 else 'Low' for p in predictions],
        'Actual': y_test[:10]
    })

    print(prediction_df.to_string(index=False))

    # Segment analysis
    print("\n" + "=" * 80)
    print("PROPENSITY DISTRIBUTION BY SEGMENT")
    print("=" * 80)

    all_predictions = trainer.models[best_model].predict(X_test) if best_model == 'lightgbm' else \
                      trainer.models[best_model].predict_proba(X_test)[:, 1]

    segments = {
        'Very High (≥0.8)': (all_predictions >= 0.8).sum(),
        'High (0.6-0.8)': ((all_predictions >= 0.6) & (all_predictions < 0.8)).sum(),
        'Medium (0.4-0.6)': ((all_predictions >= 0.4) & (all_predictions < 0.6)).sum(),
        'Low (0.2-0.4)': ((all_predictions >= 0.2) & (all_predictions < 0.4)).sum(),
        'Very Low (<0.2)': (all_predictions < 0.2).sum()
    }

    segment_df = pd.DataFrame({
        'Segment': segments.keys(),
        'Count': segments.values(),
        'Percentage': [f"{v/len(all_predictions)*100:.1f}%" for v in segments.values()]
    })

    print(segment_df.to_string(index=False))

    # Business recommendations
    print("\n" + "=" * 80)
    print("BUSINESS RECOMMENDATIONS")
    print("=" * 80)

    high_propensity = (all_predictions >= 0.6).sum()
    medium_propensity = ((all_predictions >= 0.4) & (all_predictions < 0.6)).sum()
    low_propensity = (all_predictions < 0.4).sum()

    print(f"""
📊 Customer Segments:
   • High Propensity (≥60%):    {high_propensity:5,} customers ({high_propensity/len(all_predictions)*100:5.1f}%)
   • Medium Propensity (40-60%): {medium_propensity:5,} customers ({medium_propensity/len(all_predictions)*100:5.1f}%)
   • Low Propensity (<40%):      {low_propensity:5,} customers ({low_propensity/len(all_predictions)*100:5.1f}%)

💡 Recommended Actions:
   1. High Propensity Customers:
      → Target with premium offers and personalized outreach
      → Expected conversion rate: 70-90%
      → Priority: Immediate engagement

   2. Medium Propensity Customers:
      → Nurture with targeted campaigns
      → Offer incentives and discounts
      → Expected conversion rate: 40-60%

   3. Low Propensity Customers:
      → Generic marketing campaigns
      → Focus on brand awareness
      → Reduce marketing spend for cost efficiency
    """)

    # Summary statistics
    print("=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)

    print(f"""
✓ Generated {len(customer_data):,} synthetic customers
✓ Engineered {len(feature_names)} features
✓ Trained {len(trainer.models)} models
✓ Best model: {best_model.upper()}
✓ Test AUC: {test_results[best_model]['auc']:.4f}
✓ Models saved to: models/
✓ Plots saved to: plots/

📁 Files Created:
   • data/raw/customer_data.parquet
   • data/processed/customer_features.parquet
   • models/*.pkl, models/*.txt
   • plots/{best_model}/*.png

🚀 Next Steps:
   1. Start prediction API: uvicorn src.serving.prediction_service:app --reload
   2. View notebooks for detailed analysis
   3. Customize model hyperparameters in config/model_config.yaml
   4. Deploy to production with monitoring
    """)

    print("=" * 80)
    print("✓ DEMO COMPLETE!")
    print("=" * 80)

    return trainer, test_results


if __name__ == "__main__":
    try:
        trainer, results = run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
