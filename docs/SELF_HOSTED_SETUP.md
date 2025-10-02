# Self-Hosted MLflow Server Setup

This guide shows how to set up a self-hosted MLflow server on a VM for learning and development.

## 🚀 Quick Start

### Option 1: Direct Setup

1. **Start the MLflow server:**
   ```bash
   ./scripts/start_mlflow_server.sh
   ```

2. **Access the UI:**
   - Local: http://localhost:5001
   - External: http://YOUR_VM_IP:5001

### Option 2: Docker Setup

1. **Start with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

2. **Check status:**
   ```bash
   docker-compose ps
   ```

## 🖥️ VM Setup Instructions

### 1. Create a VM
- **Provider**: AWS EC2, Google Cloud, Azure, or DigitalOcean
- **Size**: t2.micro or t3.small (sufficient for learning)
- **OS**: Ubuntu 20.04+ or similar Linux distribution

### 2. Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip git -y

# Install MLflow
pip3 install mlflow==2.14.3

# Clone your repository
git clone https://github.com/sampatsi/mlops-starter.git
cd mlops-starter
```

### 3. Configure Security

**Important: Secure your VM before exposing it!**

```bash
# Configure firewall
sudo ufw allow ssh
sudo ufw allow 5001
sudo ufw enable

# Create a non-root user
sudo adduser mlflow-user
sudo usermod -aG sudo mlflow-user
```

### 4. Start MLflow Server

```bash
# Set environment variables
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Start server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./artifacts \
  --host 0.0.0.0 \
  --port 5001
```

### 5. Configure GitHub Actions

Update your GitHub Actions workflow to use the self-hosted server:

```yaml
- name: Set MLflow Tracking URI
  run: echo "MLFLOW_TRACKING_URI=http://YOUR_VM_IP:5001" >> $GITHUB_ENV
```

## 🔧 Configuration Files

### Environment Variables
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_BACKEND_URI=sqlite:///mlflow.db
export MLFLOW_ARTIFACTS_PATH=./artifacts
```

### Python Configuration
```python
from mlflow_config import setup_self_hosted_mlflow
setup_self_hosted_mlflow()
```

## 🔒 Security Considerations

1. **Firewall**: Only open necessary ports (22 for SSH, 5001 for MLflow)
2. **Authentication**: Consider adding MLflow authentication for production
3. **HTTPS**: Use a reverse proxy (nginx) with SSL certificates
4. **Backup**: Regularly backup the SQLite database and artifacts

## 📊 Monitoring

- **Server Status**: Check if MLflow server is running
- **Disk Space**: Monitor artifacts storage usage
- **Logs**: Check MLflow server logs for errors

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   sudo lsof -ti:5001 | xargs kill -9
   ```

2. **Permission denied:**
   ```bash
   sudo chown -R $USER:$USER ./artifacts
   ```

3. **Database locked:**
   ```bash
   # Stop server and restart
   pkill -f mlflow
   ```

## 📈 Next Steps

1. **Production Setup**: Use PostgreSQL instead of SQLite
2. **Authentication**: Add MLflow authentication
3. **Monitoring**: Set up proper logging and monitoring
4. **Backup**: Implement automated backup strategy
5. **Scaling**: Consider using MLflow on Kubernetes

## 🔗 Useful Commands

```bash
# Check MLflow server status
curl http://localhost:5001/health

# View server logs
tail -f mlflow.log

# Backup database
cp mlflow.db mlflow_backup_$(date +%Y%m%d).db

# Restore database
cp mlflow_backup_20240101.db mlflow.db
```
