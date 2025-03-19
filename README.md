## Poject Overview
This project aims to predict customer churn using machine learning models like XGBoost. The trained model is deployed as REST API using FastAPI, containerized with docker, and deployed to AWS Lambda using a container image.The syatem also stores logs and data in AWS S3 and cloudwatch for monitoring and analysis.
## How to Run:
### 1. Train the Model:
```bash
python train_model.py
