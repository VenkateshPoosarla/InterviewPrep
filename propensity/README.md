# Propensity Modeling System
## Complete End-to-End ML Pipeline for Customer Conversion Prediction

**Purpose:** Predict the likelihood (propensity) that a customer will convert (purchase, subscribe, upgrade) based on their behavior and characteristics.

**Business Use Cases:**
- E-commerce: Purchase propensity
- SaaS: Upgrade/subscription propensity
- Banking: Loan approval propensity
- Marketing: Campaign response propensity

**System Scale:** Production-ready ML pipeline handling millions of predictions daily

---

## Project Structure

```
propensity/
├── README.md                          # This file
├── data/
│   ├── raw/                           # Raw customer data
│   ├── processed/                     # Cleaned and feature-engineered data
│   └── sample_generator.py            # Generate synthetic data
├── src/
│   ├── data_pipeline/
│   │   ├── data_loader.py            # Load and validate data
│   │   └── feature_engineering.py    # Create features
│   ├── models/
│   │   ├── propensity_model.py       # Main model training
│   │   ├── model_evaluation.py       # Evaluation metrics
│   │   └── model_explainability.py   # SHAP, feature importance
│   ├── serving/
│   │   ├── prediction_service.py     # FastAPI serving
│   │   └── batch_scoring.py          # Batch predictions
│   └── monitoring/
│       ├── model_monitoring.py       # Drift detection, performance
│       └── ab_testing.py             # A/B test framework
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA
│   ├── 02_feature_engineering.ipynb  # Feature analysis
│   ├── 03_model_training.ipynb       # Model experiments
│   └── 04_model_evaluation.ipynb     # Results analysis
├── config/
│   ├── data_config.yaml              # Data pipeline config
│   ├── model_config.yaml             # Model hyperparameters
│   └── serving_config.yaml           # Serving configuration
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   └── test_serving.py
├── mlflow/                            # MLflow tracking
├── models/                            # Saved models
├── requirements.txt
└── demo.py                            # End-to-end demo
```

---

## System Flow

```
Customer Data
    ↓
Data Pipeline (Clean, Validate, Engineer Features)
    ↓
Model Training (LightGBM, Logistic Regression, Neural Network)
    ↓
Model Evaluation (AUC, Precision, Recall, Calibration)
    ↓
Model Explainability (SHAP values, Feature Importance)
    ↓
Model Deployment (FastAPI API + Batch Scoring)
    ↓
Monitoring (Drift Detection, Performance Tracking)
    ↓
Predictions (Real-time API or Batch)
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python data/sample_generator.py

# Run end-to-end demo
python demo.py

# Start prediction API
uvicorn src.serving.prediction_service:app --reload --port 8000

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "12345",
    "age": 35,
    "tenure_months": 24,
    "total_purchases": 15,
    "avg_order_value": 75.50,
    "days_since_last_purchase": 7,
    "email_open_rate": 0.65,
    "website_visits_30d": 12
  }'
```

---

## Key Features

1. **Complete Data Pipeline**
   - Synthetic data generation for demo
   - Data validation and quality checks
   - Feature engineering (RFM, behavioral, demographic)
   - Train/validation/test splits

2. **Multiple Model Approaches**
   - Logistic Regression (baseline)
   - LightGBM (production standard)
   - Neural Network (deep learning)
   - Ensemble methods

3. **Model Explainability**
   - SHAP values for feature importance
   - Partial dependence plots
   - Individual prediction explanations

4. **Production Serving**
   - Real-time API (FastAPI)
   - Batch scoring pipeline
   - Model versioning
   - A/B testing framework

5. **Monitoring & Observability**
   - Data drift detection (PSI)
   - Model performance tracking
   - Prediction distribution monitoring
   - Alerting on degradation

6. **Business Metrics**
   - Conversion lift analysis
   - ROI calculations
   - Segment-level performance
   - Calibration analysis

---

## Business Value

**Problem:**
- Only 2-5% of customers convert naturally
- Marketing budget wasted on low-propensity customers
- High-propensity customers not prioritized

**Solution:**
- Predict conversion probability for each customer
- Target high-propensity customers with personalized offers
- Reduce marketing spend on low-propensity segments

**Impact:**
- 30-50% increase in conversion rate
- 20-40% reduction in marketing costs
- 3-5x improvement in ROI
