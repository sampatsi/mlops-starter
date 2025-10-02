# GitHub Actions Setup for Automatic Model Registration

This guide shows how to configure GitHub Actions to automatically train and register models when code is pushed to the repository.

## 🔧 Configuration

### 1. GitHub Secrets Setup

Go to your repository settings and add these secrets:

**Required Secrets:**
- `MLFLOW_TRACKING_URI`: Your MLflow server URL
  - For self-hosted: `http://YOUR_VM_IP:5001`
  - For local development: `http://localhost:5001`

**Optional Secrets (with defaults):**
- `MLFLOW_BACKEND_URI`: `sqlite:///mlflow.db`
- `MLFLOW_EXPERIMENT`: `iris-classification-ci`
- `MLFLOW_MODEL_NAME`: `iris-classifier`

### 2. How to Add Secrets

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret with its value

### 3. Self-Hosted MLflow Server

If using a self-hosted server, make sure:

1. **Server is accessible** from GitHub Actions runners
2. **Firewall allows** port 5001
3. **CORS is enabled** (if needed)
4. **Authentication** is configured (for production)

## 🚀 How It Works

### Automatic Triggers

The CI pipeline runs when:
- ✅ **Push to main/master branch**
- ✅ **Pull requests** (for testing)
- ✅ **Changes to src/** files
- ✅ **Changes to requirements.txt**

### Pipeline Steps

1. **Checkout Code** - Gets the latest code
2. **Setup Python** - Installs Python 3.11
3. **Install Dependencies** - Installs requirements.txt
4. **Lint Code** - Runs flake8 for code quality
5. **Run Tests** - Executes pytest tests
6. **Train + Evaluate** - Trains model with basic parameters
7. **Register Model** - Registers model in MLflow (main branch only)
8. **Train Best Model** - Trains multiple models and picks the best one
9. **Upload Artifacts** - Saves model artifacts and metadata

### Model Registration

When you push to main/master:
- 🎯 **Multiple hyperparameters** are tested
- 🏆 **Best model** is automatically selected
- 📦 **Model is registered** in MLflow Model Registry
- 📊 **Metrics and metadata** are logged
- 🔄 **Artifacts are uploaded** to GitHub Actions

## 📊 Viewing Results

### MLflow UI
- **URL**: Your `MLFLOW_TRACKING_URI` secret value
- **Models**: Check the "Models" section
- **Experiments**: View training runs and metrics

### GitHub Actions
- **URL**: `https://github.com/USERNAME/REPO/actions`
- **Artifacts**: Download model artifacts and metadata
- **Logs**: View detailed training logs

## 🔧 Customization

### Environment Variables

You can customize the pipeline by setting these environment variables in your repository secrets:

```yaml
# Required
MLFLOW_TRACKING_URI: "http://your-mlflow-server:5001"

# Optional
MLFLOW_BACKEND_URI: "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT: "my-custom-experiment"
MLFLOW_MODEL_NAME: "my-custom-model"
```

### Hyperparameters

Edit `scripts/ci_train_model.py` to modify the hyperparameter combinations:

```python
hyperparams_combinations = [
    {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
    {'n_estimators': 150, 'max_depth': 3, 'random_state': 42},
    {'n_estimators': 200, 'max_depth': 4, 'random_state': 42},
    # Add your own combinations
]
```

## 🚨 Troubleshooting

### Common Issues

1. **MLflow server not accessible**
   - Check if server is running
   - Verify firewall settings
   - Test connectivity from GitHub Actions

2. **Model registration fails**
   - Check MLflow server logs
   - Verify model name doesn't conflict
   - Ensure proper permissions

3. **Tests fail**
   - Check Python version compatibility
   - Verify all dependencies are installed
   - Review test logs for specific errors

### Debug Mode

To debug locally, run:

```bash
# Set environment variables
export MLFLOW_TRACKING_URI="http://localhost:5001"
export MLFLOW_BACKEND_URI="sqlite:///mlflow.db"
export MLFLOW_EXPERIMENT="iris-classification-ci"
export MLFLOW_MODEL_NAME="iris-classifier"

# Run the CI script
python scripts/ci_train_model.py
```

## 📈 Monitoring

### Success Indicators

- ✅ **Green checkmark** in GitHub Actions
- ✅ **Model appears** in MLflow Model Registry
- ✅ **Artifacts uploaded** successfully
- ✅ **Metadata file** created

### Failure Indicators

- ❌ **Red X** in GitHub Actions
- ❌ **Error logs** in GitHub Actions
- ❌ **No model** in MLflow registry
- ❌ **Missing artifacts**

## 🔄 Next Steps

1. **Set up secrets** in GitHub repository
2. **Push code** to trigger the pipeline
3. **Monitor** GitHub Actions for success
4. **Check** MLflow UI for registered models
5. **Download** artifacts for deployment

Your models will now be automatically trained and registered every time you push code! 🚀
