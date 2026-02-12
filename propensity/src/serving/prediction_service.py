"""
Production Prediction Service

FastAPI service for real-time propensity score predictions.
Handles single predictions and batch scoring.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import torch
import torch.nn as nn
from datetime import datetime
import json
import os
from pathlib import Path

# Import feature engineering
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_pipeline.feature_engineering import PropensityFeatureEngineering


app = FastAPI(
    title="Propensity Prediction Service",
    description="Real-time customer conversion propensity predictions",
    version="1.0.0"
)


# Request/Response Models
class CustomerData(BaseModel):
    """Single customer data for prediction"""
    customer_id: str
    age: int = Field(..., ge=18, le=100)
    gender: str
    income_bracket: str
    education: str
    location_type: str
    tenure_months: int = Field(..., ge=0)
    total_purchases: int = Field(..., ge=0)
    avg_order_value: float = Field(..., ge=0)
    total_revenue: float = Field(..., ge=0)
    days_since_last_purchase: int = Field(..., ge=0)
    purchase_frequency: float = Field(..., ge=0)
    is_repeat_customer: int = Field(..., ge=0, le=1)
    email_sent_30d: int = Field(..., ge=0)
    email_opened_30d: int = Field(..., ge=0)
    email_clicked_30d: int = Field(..., ge=0)
    email_open_rate: float = Field(..., ge=0, le=1)
    email_click_rate: float = Field(..., ge=0, le=1)
    website_visits_30d: int = Field(..., ge=0)
    avg_session_duration_min: float = Field(..., ge=0)
    pages_per_session: float = Field(..., ge=0)
    has_mobile_app: int = Field(..., ge=0, le=1)
    app_sessions_30d: int = Field(..., ge=0)
    social_media_follower: int = Field(..., ge=0, le=1)
    num_categories_purchased: int = Field(..., ge=0)
    purchases_with_discount_pct: float = Field(..., ge=0, le=1)
    cart_abandonment_rate: float = Field(..., ge=0, le=1)
    product_views_30d: int = Field(..., ge=0)
    wishlist_items: int = Field(..., ge=0)
    return_rate: float = Field(..., ge=0, le=1)


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: List[CustomerData]


class PredictionResponse(BaseModel):
    """Prediction response"""
    customer_id: str
    propensity_score: float
    risk_category: str
    confidence: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_count: int


# Model Manager
class ModelManager:
    """Manages loaded models and feature engineering"""

    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.model_type = None
        self.feature_engineer = PropensityFeatureEngineering()
        self.feature_names = None
        self.metadata = None

    def load_latest_model(self, model_type='lightgbm'):
        """Load the most recent model"""
        print(f"Loading latest {model_type} model...")

        # Find latest model file
        model_files = list(Path(self.model_dir).glob(f'{model_type}_*.txt' if model_type == 'lightgbm' else f'{model_type}_*.pkl'))

        if not model_files:
            raise FileNotFoundError(f"No {model_type} model found in {self.model_dir}")

        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

        # Load model
        if model_type == 'lightgbm':
            self.model = lgb.Booster(model_file=str(latest_model))
        else:
            self.model = joblib.load(latest_model)

        self.model_type = model_type

        # Load metadata
        metadata_files = list(Path(self.model_dir).glob('model_metadata_*.json'))
        if metadata_files:
            latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)
            with open(latest_metadata) as f:
                self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', [])

        print(f"✓ Loaded model: {latest_model.name}")
        return self

    def prepare_features(self, customer_data: Dict) -> np.ndarray:
        """Prepare features from raw customer data"""
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])

        # Engineer features
        df_features = self.feature_engineer.create_features(df)

        # Select features used in training
        if self.feature_names:
            feature_cols = [col for col in self.feature_names if col in df_features.columns]
            X = df_features[feature_cols].values
        else:
            # Use all numeric features except ID and target
            feature_cols = df_features.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in feature_cols if col not in ['customer_id', 'converted']]
            X = df_features[feature_cols].values

        return X

    def predict(self, customer_data: Dict) -> float:
        """Make single prediction"""
        X = self.prepare_features(customer_data)

        if self.model_type == 'lightgbm':
            propensity = self.model.predict(X)[0]
        elif self.model_type == 'neural_network':
            with torch.no_grad():
                propensity = self.model(torch.FloatTensor(X)).item()
        else:
            propensity = self.model.predict_proba(X)[0, 1]

        return float(propensity)

    def predict_batch(self, customers_data: List[Dict]) -> np.ndarray:
        """Make batch predictions"""
        # Prepare all features
        df = pd.DataFrame(customers_data)
        df_features = self.feature_engineer.create_features(df)

        if self.feature_names:
            feature_cols = [col for col in self.feature_names if col in df_features.columns]
            X = df_features[feature_cols].values
        else:
            feature_cols = df_features.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in feature_cols if col not in ['customer_id', 'converted']]
            X = df_features[feature_cols].values

        # Predict
        if self.model_type == 'lightgbm':
            propensities = self.model.predict(X)
        elif self.model_type == 'neural_network':
            with torch.no_grad():
                propensities = self.model(torch.FloatTensor(X)).numpy()
        else:
            propensities = self.model.predict_proba(X)[:, 1]

        return propensities


# Initialize model manager
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        model_manager.load_latest_model(model_type='lightgbm')
        print("✓ Prediction service ready")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Service will start but predictions will fail until model is loaded")


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Propensity Prediction Service",
        "model_loaded": model_manager.model is not None,
        "model_type": model_manager.model_type,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
def get_model_info():
    """Get loaded model information"""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="No model loaded")

    return {
        "model_type": model_manager.model_type,
        "num_features": len(model_manager.feature_names) if model_manager.feature_names else None,
        "metadata": model_manager.metadata
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(customer: CustomerData):
    """
    Predict conversion propensity for a single customer

    Returns propensity score between 0 and 1
    """
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="No model loaded")

    try:
        # Convert to dict
        customer_dict = customer.dict()

        # Predict
        propensity_score = model_manager.predict(customer_dict)

        # Categorize risk
        if propensity_score >= 0.7:
            risk_category = "High"
        elif propensity_score >= 0.4:
            risk_category = "Medium"
        else:
            risk_category = "Low"

        # Confidence (simple heuristic based on score extremity)
        confidence = abs(propensity_score - 0.5) * 2

        return PredictionResponse(
            customer_id=customer.customer_id,
            propensity_score=round(propensity_score, 4),
            risk_category=risk_category,
            confidence=round(confidence, 4),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Predict conversion propensity for multiple customers

    Optimized for batch processing
    """
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="No model loaded")

    try:
        # Convert to list of dicts
        customers_data = [c.dict() for c in request.customers]

        # Batch predict
        propensity_scores = model_manager.predict_batch(customers_data)

        # Build responses
        predictions = []
        for customer, score in zip(request.customers, propensity_scores):
            # Categorize
            if score >= 0.7:
                risk_category = "High"
            elif score >= 0.4:
                risk_category = "Medium"
            else:
                risk_category = "Low"

            confidence = abs(score - 0.5) * 2

            predictions.append(PredictionResponse(
                customer_id=customer.customer_id,
                propensity_score=round(float(score), 4),
                risk_category=risk_category,
                confidence=round(float(confidence), 4),
                timestamp=datetime.now().isoformat()
            ))

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/model/reload")
def reload_model(model_type: str = 'lightgbm'):
    """Reload the model (for model updates)"""
    try:
        model_manager.load_latest_model(model_type=model_type)
        return {
            "status": "success",
            "message": f"Reloaded {model_type} model",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
