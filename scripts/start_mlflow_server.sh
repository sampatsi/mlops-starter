#!/bin/bash

# Self-hosted MLflow Server Startup Script
# This script sets up MLflow with SQLite backend and local artifacts storage

echo "🚀 Starting self-hosted MLflow server..."

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Create artifacts directory if it doesn't exist
mkdir -p ./artifacts

# Start MLflow server
echo "📊 MLflow server starting with:"
echo "   - Backend Store: SQLite (mlflow.db)"
echo "   - Artifacts: ./artifacts"
echo "   - Host: 0.0.0.0"
echo "   - Port: 5001"
echo ""
echo "🌐 Access MLflow UI at: http://localhost:5001"
echo "🔗 For external access: http://YOUR_VM_IP:5001"
echo ""
echo "Press Ctrl+C to stop the server"
echo "----------------------------------------"

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./artifacts \
  --host 0.0.0.0 \
  --port 5001
