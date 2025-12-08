# MLOps Fraud Detection Demo Script

## 🎯 Introduction (1 min)

Welcome to the demonstration of our MLOps Fraud Detection project. This system automatically detects fraudulent credit card transactions using machine learning techniques with a complete production-ready pipeline.

**Business Problem**: Credit card fraud costs billions annually. Manual review is impossible at scale, so we need an automated system that can detect fraud in real-time with high accuracy.

## 🏗️ Architecture Overview (1 min)

Our system consists of several interconnected components:

1. **Data Pipeline**: DVC for data versioning and preprocessing
2. **Model Training**: MLflow for experiment tracking and model registry
3. **API Service**: FastAPI for real-time predictions
4. **User Interface**: Streamlit dashboard for monitoring and interaction
5. **Monitoring**: Evidently AI for drift detection
6. **Deployment**: Docker containers with CI/CD to Railway

## 🔗 Pipeline Demonstration (2 min)

### DVC Pipeline
```bash
# Show the DVC pipeline
cat dvc.yaml

# Reproduce the pipeline
dvc repro
```

### MLflow Experiment Tracking
```bash
# Show MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

Navigate to http://localhost:5000 to show:
- Experiment tracking
- Model versions
- Parameter comparison

## 💻 Code Walkthrough (2 min)

### Training Code
```bash
# Show training script
cat src/train.py
```

Key features:
- Logistic Regression model
- MLflow integration for experiment tracking
- ROC-AUC and PR-AUC metrics logging

### Dockerfile
```bash
# Show API Dockerfile
cat docker/api.Dockerfile
```

Key aspects:
- Python 3.11 slim base image
- Dependency installation
- Proper environment setup

## 🧪 Testing (1 min)

### Pytest Results
```bash
# Run tests
pytest tests/ -v
```

Show test coverage:
- API endpoint tests
- Data drift detection tests
- Model drift detection tests
- Request logging tests

## 🔄 CI/CD Pipeline (1 min)

### GitHub Actions
Show `.github/workflows/ci.yml`:
- Multi-Python version testing
- Code quality checks with flake8
- Docker image building and testing
- Automatic publishing to Docker Hub

Show `.github/workflows/cd.yml`:
- Railway deployment on push to main/dev-clean
- Staging and production environments

## ☁️ Railway Deployment (1 min)

### Health Check
```bash
# Check API health
curl -X GET https://fraud-api-production.up.railway.app/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "n_features_expected": 30
}
```

### Prediction Example
```bash
# Make a prediction
curl -X POST https://fraud-api-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 
                 1.1, -1.2, 1.3, -1.4, 1.5, -1.6, 1.7, -1.8, 1.9, -2.0, 
                 2.1, -2.2, 2.3, -2.4, 2.5, -2.6, 2.7, -2.8, 2.9, -3.0]
  }'
```

Expected response:
```json
{
  "prediction": 0,
  "probability": 0.15
}
```

## 🖥️ Streamlit UI Demo (1 min)

Access the UI at https://fraud-ui-production.up.railway.app

Demonstrate:
1. **Prediction Tab**: Enter features and get real-time fraud prediction
2. **Monitoring Tab**: View total predictions, fraud rate, and average probability
3. **Data Drift Tab**: Run drift analysis and visualize feature changes
4. **Model Drift Tab**: Monitor model performance degradation

## 📊 Evidently Monitoring (1 min)

### Data Drift Detection
Show drift detection capabilities:
- Kolmogorov-Smirnov tests for each feature
- Interactive visualizations of drift metrics
- Automated alerts for significant changes

### Model Quality Monitoring
Show model performance tracking:
- ROC-AUC score comparison
- Fraud prediction rate changes
- Probability distribution analysis
- Abnormal prediction detection

## 📝 Summary (1 min)

Our MLOps pipeline provides:
- **Reproducible** data and model pipelines
- **Monitored** production performance
- **Automated** CI/CD deployment
- **Scalable** Docker containerization
- **Observable** drift detection system

The system is ready for production use with comprehensive monitoring and alerting capabilities.

## 🚀 Next Steps

1. Enhance model with more sophisticated algorithms
2. Add more comprehensive monitoring dashboards
3. Implement automated retraining pipelines
4. Expand to other types of financial fraud detection