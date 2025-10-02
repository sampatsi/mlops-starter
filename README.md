
# MLOps Starter: Model Registry + CI/CD (Hands-on)

A tiny end-to-end project to learn MLOps fundamentals:
- Train a model (scikit-learn on Iris)
- Track runs in MLflow (experiment + artifacts)
- **Register** the best model in MLflow Model Registry
- Validate inputs with a schema
- CI pipeline (GitHub Actions) to: lint, test, train, evaluate, and (optionally) register

> Registry runs locally by default with MLflow's file backend. You can later switch to a real tracking server (e.g., sqlite + artifact store, S3/MinIO) without changing code.

## Quickstart

### 0) Prereqs
- Python 3.10+
- `pip install -r requirements.txt`

### 1) Local MLflow Tracking
- Start UI (optional):  
  ```bash
  mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000
  ```
- The default tracking URI is local folder `mlruns/` (set in code).

### 2) Train & Evaluate
```bash
python src/train.py --n_estimators 150 --max_depth 3
python src/evaluate.py
```

### 3) Register the Best Model
```bash
python src/register.py --metric f1 --min-improve 0.0 --model-name iris_classifier
```
This promotes the latest run's model to the **Model Registry** (creating `iris_classifier` if not exists).

### 4) Inference (with input validation)
```bash
python src/predict.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2
```
Loads the **latest** registered model and returns the class prediction.

### 5) CI: GitHub Actions
- On every push/PR:
  - Lint + unit tests
  - Train + evaluate
  - (Optionally) register on `main` branch only

Create a repo and add this folder. Then push to GitHub.

## Structure
```
mlops-starter/
  ├─ src/
  │   ├─ data.py
  │   ├─ schema.py
  │   ├─ train.py
  │   ├─ evaluate.py
  │   ├─ register.py
  │   └─ predict.py
  ├─ tests/
  │   └─ test_schema.py
  ├─ .github/workflows/ci.yml
  ├─ requirements.txt
  ├─ README.md
  └─ .gitignore
```

## Switch to a Real Registry Later
- Set `MLFLOW_TRACKING_URI` to a server, e.g. sqlite:
  ```bash
  export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
  mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./artifacts --host 0.0.0.0 --port 5001
  ```
- Update env var in CI and local shells.

Happy shipping!
