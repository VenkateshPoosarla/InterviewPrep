"""
Propensity Model Training

Trains multiple models to predict customer conversion probability:
1. Logistic Regression (baseline)
2. LightGBM (production standard)
3. Neural Network (deep learning)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import json
from datetime import datetime


class PropensityDataset(Dataset):
    """PyTorch dataset for propensity modeling"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PropensityNN(nn.Module):
    """Neural network for propensity prediction"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()


class PropensityModelTrainer:
    """Train and evaluate propensity models"""

    def __init__(self):
        self.models = {}
        self.feature_names = None
        self.metrics = {}

    def prepare_data(self, df, target_col='converted', test_size=0.2, val_size=0.1):
        """
        Prepare train/val/test splits

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Preparing data...")

        # Separate features and target
        feature_cols = [col for col in df.columns
                       if col not in ['customer_id', target_col]]
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df[target_col].values

        # Train/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train/val split
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=42, stratify=y_temp
        )

        print(f"✓ Train: {len(X_train):,} ({y_train.mean():.2%} positive)")
        print(f"✓ Val:   {len(X_val):,} ({y_val.mean():.2%} positive)")
        print(f"✓ Test:  {len(X_test):,} ({y_test.mean():.2%} positive)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train logistic regression baseline"""
        print("\nTraining Logistic Regression...")

        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate
        train_proba = model.predict_proba(X_train)[:, 1]
        val_proba = model.predict_proba(X_val)[:, 1]

        train_auc = roc_auc_score(y_train, train_proba)
        val_auc = roc_auc_score(y_val, val_proba)

        print(f"✓ Train AUC: {train_auc:.4f}")
        print(f"✓ Val AUC:   {val_auc:.4f}")

        self.models['logistic_regression'] = model
        self.metrics['logistic_regression'] = {
            'train_auc': train_auc,
            'val_auc': val_auc
        }

        return model

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model (production standard)"""
        print("\nTraining LightGBM...")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_data_in_leaf': 100,
            'verbose': -1,
            'seed': 42
        }

        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )

        # Evaluate
        train_proba = model.predict(X_train)
        val_proba = model.predict(X_val)

        train_auc = roc_auc_score(y_train, train_proba)
        val_auc = roc_auc_score(y_val, val_proba)

        print(f"✓ Train AUC: {train_auc:.4f}")
        print(f"✓ Val AUC:   {val_auc:.4f}")
        print(f"✓ Best iteration: {model.best_iteration}")

        self.models['lightgbm'] = model
        self.metrics['lightgbm'] = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'n_estimators': model.best_iteration
        }

        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Important Features:")
        print(feature_importance.head(10).to_string(index=False))

        return model

    def train_neural_network(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train neural network model"""
        print("\nTraining Neural Network...")

        # Create datasets
        train_dataset = PropensityDataset(X_train, y_train)
        val_dataset = PropensityDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Create model
        input_dim = X_train.shape[1]
        model = PropensityNN(input_dim, hidden_dims=[128, 64, 32])

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        best_val_auc = 0
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate
            model.eval()
            val_preds = []
            val_true = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    val_preds.extend(outputs.numpy())
                    val_true.extend(y_batch.numpy())

            val_auc = roc_auc_score(val_true, val_preds)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val AUC: {val_auc:.4f}")

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        model.load_state_dict(best_model_state)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_proba = model(torch.FloatTensor(X_train)).numpy()
            val_proba = model(torch.FloatTensor(X_val)).numpy()

        train_auc = roc_auc_score(y_train, train_proba)
        val_auc = roc_auc_score(y_val, val_proba)

        print(f"✓ Train AUC: {train_auc:.4f}")
        print(f"✓ Val AUC:   {val_auc:.4f}")

        self.models['neural_network'] = model
        self.metrics['neural_network'] = {
            'train_auc': train_auc,
            'val_auc': val_auc
        }

        return model

    def evaluate_on_test(self, X_test, y_test):
        """Evaluate all models on test set"""
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)

        results = {}

        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()}:")

            # Get predictions
            if model_name == 'neural_network':
                model.eval()
                with torch.no_grad():
                    y_proba = model(torch.FloatTensor(X_test)).numpy()
            elif model_name == 'lightgbm':
                y_proba = model.predict(X_test)
            else:
                y_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            test_auc = roc_auc_score(y_test, y_proba)
            y_pred = (y_proba >= 0.5).astype(int)

            print(f"  AUC: {test_auc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Not Converted', 'Converted']))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
            print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

            results[model_name] = {
                'auc': test_auc,
                'predictions': y_proba
            }

        return results

    def save_models(self, output_dir='models'):
        """Save trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for model_name, model in self.models.items():
            if model_name == 'neural_network':
                torch.save(model.state_dict(), f'{output_dir}/{model_name}_{timestamp}.pth')
            elif model_name == 'lightgbm':
                model.save_model(f'{output_dir}/{model_name}_{timestamp}.txt')
            else:
                joblib.dump(model, f'{output_dir}/{model_name}_{timestamp}.pkl')

        # Save feature names and metrics
        metadata = {
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'timestamp': timestamp
        }

        with open(f'{output_dir}/model_metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Models saved to {output_dir}/")


def train_propensity_models(data_path='data/processed/customer_features.parquet'):
    """
    Main training pipeline

    Trains all propensity models and evaluates them
    """
    print("="*60)
    print("PROPENSITY MODEL TRAINING")
    print("="*60)

    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df):,} customers")

    # Initialize trainer
    trainer = PropensityModelTrainer()

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)

    # Train models
    trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
    trainer.train_lightgbm(X_train, y_train, X_val, y_val)
    trainer.train_neural_network(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    test_results = trainer.evaluate_on_test(X_test, y_test)

    # Save models
    trainer.save_models()

    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)

    return trainer, test_results


if __name__ == "__main__":
    trainer, results = train_propensity_models()
