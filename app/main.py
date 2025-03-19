from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("model/churn_model.pkl")

# Define input schema
class CustomerData(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return {"churn_prediction": bool(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
