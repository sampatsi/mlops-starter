#!/usr/bin/env python3
"""
Train and register a model that can be picked up from Git repository.
This script trains a model and registers it with MLflow for easy deployment.
"""

import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import load_data


def train_and_register_model():
    """Train a model and register it with MLflow."""
    
    # Set MLflow tracking URI to the self-hosted server
    mlflow.set_tracking_uri("http://localhost:5001")
    
    # Create experiment
    experiment_name = "iris-classification"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Created experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"✅ Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    # Load data
    X_train, X_test, y_train, y_test, target_names = load_data()
    
    with mlflow.start_run() as run:
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        })
        
        # Log metrics
        mlflow.log_metrics({
            "f1_score": f1,
            "accuracy": accuracy
        })
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="iris-classifier"
        )
        
        print(f"✅ Model trained and registered!")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Model Name: iris-classifier")
        
        return run.info.run_id


def load_model_from_git():
    """Load the registered model for inference."""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5001")
    
    # Load the latest version of the registered model
    model_name = "iris-classifier"
    model_version = "latest"
    
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded from Git/MLflow: {model_uri}")
        
        # Test the model
        X_train, X_test, y_train, y_test, target_names = load_data()
        predictions = model.predict(X_test[:5])
        
        print(f"✅ Model test predictions: {predictions}")
        print(f"   Target names: {target_names}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


if __name__ == "__main__":
    print("🚀 Training and registering model for Git pickup...")
    print("=" * 50)
    
    # Train and register model
    run_id = train_and_register_model()
    
    print("\n" + "=" * 50)
    print("📦 Loading model from Git/MLflow...")
    
    # Load model from Git/MLflow
    model = load_model_from_git()
    
    if model:
        print("\n✅ Success! Model can be picked up from Git repository via MLflow!")
        print("🌐 Access MLflow UI at: http://localhost:5001")
        print("📊 View registered models in the Models section")
    else:
        print("\n❌ Failed to load model from Git/MLflow")
