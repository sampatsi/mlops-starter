#!/usr/bin/env python3
"""
CI/CD Model Training Script
This script is designed to run in GitHub Actions and automatically
train and register models when code is pushed to the repository.
"""

import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import load_data


def get_mlflow_config():
    """Get MLflow configuration from environment variables."""
    return {
        'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001'),
        'backend_uri': os.getenv('MLFLOW_BACKEND_URI', 'sqlite:///mlflow.db'),
        'experiment_name': os.getenv('MLFLOW_EXPERIMENT', 'iris-classification-ci'),
        'model_name': os.getenv('MLFLOW_MODEL_NAME', 'iris-classifier'),
    }


def train_model_with_hyperparameters():
    """Train model with different hyperparameters for better performance."""
    
    config = get_mlflow_config()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config['tracking_uri'])
    print(f"🔗 MLflow Tracking URI: {config['tracking_uri']}")
    
    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(config['experiment_name'])
        print(f"✅ Created experiment: {config['experiment_name']}")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(config['experiment_name'])
        experiment_id = experiment.experiment_id
        print(f"✅ Using existing experiment: {config['experiment_name']}")
    
    mlflow.set_experiment(config['experiment_name'])
    
    # Load data
    X_train, X_test, y_train, y_test, target_names = load_data()
    
    # Different hyperparameter combinations to try
    hyperparams_combinations = [
        {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
        {'n_estimators': 150, 'max_depth': 3, 'random_state': 42},
        {'n_estimators': 200, 'max_depth': 4, 'random_state': 42},
    ]
    
    best_model = None
    best_f1 = 0
    best_run_id = None
    
    for i, params in enumerate(hyperparams_combinations):
        with mlflow.start_run(run_name=f"run_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred, average='macro')
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics({
                'f1_score': f1,
                'accuracy': accuracy
            })
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=config['model_name']
            )
            
            print(f"✅ Run {i+1}: F1={f1:.4f}, Accuracy={accuracy:.4f}")
            
            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_run_id = mlflow.active_run().info.run_id
    
    return best_run_id, best_f1


def register_best_model():
    """Register the best model version."""
    
    config = get_mlflow_config()
    mlflow.set_tracking_uri(config['tracking_uri'])
    
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get latest versions
        latest_versions = client.get_latest_versions(config['model_name'])
        
        if latest_versions:
            latest_version = latest_versions[0]
            print(f"✅ Latest model version: {latest_version.version}")
            print(f"   Model URI: models:/{config['model_name']}/{latest_version.version}")
            print(f"   Run ID: {latest_version.run_id}")
            print(f"   Status: {latest_version.current_stage}")
            
            # Test the model
            model_uri = f"models:/{config['model_name']}/{latest_version.version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Quick test
            X_train, X_test, y_train, y_test, target_names = load_data()
            test_predictions = model.predict(X_test[:5])
            
            print(f"✅ Model test predictions: {test_predictions}")
            print(f"   Target names: {target_names}")
            
            return True
        else:
            print("❌ No model versions found")
            return False
            
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        return False


def create_model_metadata():
    """Create metadata file for the model."""
    
    config = get_mlflow_config()
    metadata = {
        'model_name': config['model_name'],
        'experiment_name': config['experiment_name'],
        'tracking_uri': config['tracking_uri'],
        'created_at': datetime.now().isoformat(),
        'git_commit': os.getenv('GITHUB_SHA', 'local'),
        'git_branch': os.getenv('GITHUB_REF_NAME', 'local'),
        'ci_run_id': os.getenv('GITHUB_RUN_ID', 'local'),
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model metadata saved to model_metadata.json")
    return metadata


if __name__ == "__main__":
    print("🚀 Starting CI/CD Model Training Pipeline...")
    print("=" * 60)
    
    # Check if running in CI
    if os.getenv('GITHUB_ACTIONS'):
        print("🤖 Running in GitHub Actions")
        print(f"   Commit: {os.getenv('GITHUB_SHA', 'N/A')}")
        print(f"   Branch: {os.getenv('GITHUB_REF_NAME', 'N/A')}")
        print(f"   Run ID: {os.getenv('GITHUB_RUN_ID', 'N/A')}")
    else:
        print("💻 Running locally")
    
    print("\n" + "=" * 60)
    print("🎯 Training models with different hyperparameters...")
    
    # Train models
    best_run_id, best_f1 = train_model_with_hyperparameters()
    
    print(f"\n🏆 Best model: Run ID {best_run_id}, F1 Score: {best_f1:.4f}")
    
    print("\n" + "=" * 60)
    print("📦 Registering best model...")
    
    # Register best model
    success = register_best_model()
    
    print("\n" + "=" * 60)
    print("📋 Creating model metadata...")
    
    # Create metadata
    metadata = create_model_metadata()
    
    if success:
        print("\n✅ CI/CD Pipeline completed successfully!")
        print("🌐 Model is now available in MLflow Model Registry")
        print(f"📊 MLflow UI: {metadata['tracking_uri']}")
    else:
        print("\n❌ CI/CD Pipeline failed!")
        sys.exit(1)
