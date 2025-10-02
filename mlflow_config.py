"""
MLflow Configuration for Self-Hosted Server

This module provides configuration for connecting to a self-hosted MLflow server.
Use this when running MLflow on a separate VM or server.
"""

import os

# Self-hosted MLflow server configuration
MLFLOW_SERVER_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_BACKEND_URI = "sqlite:///mlflow.db"
MLFLOW_ARTIFACTS_PATH = "./artifacts"

# Environment variables for self-hosted setup
SELF_HOSTED_CONFIG = {
    "MLFLOW_TRACKING_URI": MLFLOW_SERVER_URL,
    "MLFLOW_BACKEND_URI": MLFLOW_BACKEND_URI,
    "MLFLOW_ARTIFACTS_PATH": MLFLOW_ARTIFACTS_PATH,
}

def setup_self_hosted_mlflow():
    """
    Set up environment variables for self-hosted MLflow server.
    Call this function before running any MLflow operations.
    """
    for key, value in SELF_HOSTED_CONFIG.items():
        os.environ[key] = value
    print(f"✅ MLflow configured for self-hosted server: {MLFLOW_SERVER_URL}")

def get_mlflow_server_command():
    """
    Get the command to start the MLflow server.
    """
    return f"""
mlflow server \\
  --backend-store-uri {MLFLOW_BACKEND_URI} \\
  --artifacts-destination {MLFLOW_ARTIFACTS_PATH} \\
  --host 0.0.0.0 \\
  --port 5001
"""

if __name__ == "__main__":
    print("🔧 Self-hosted MLflow Configuration")
    print("=" * 40)
    print(f"Server URL: {MLFLOW_SERVER_URL}")
    print(f"Backend URI: {MLFLOW_BACKEND_URI}")
    print(f"Artifacts Path: {MLFLOW_ARTIFACTS_PATH}")
    print("\n🚀 To start the server, run:")
    print(get_mlflow_server_command())
