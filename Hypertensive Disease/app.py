from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Hypertensive Disease Prediction API", description="API for predicting hypertensive disease using SVM model")

# Load the model and scaler
try:
    model = joblib.load("Hypertensive_Disease_SVM.pkl")
    scaler = joblib.load("Hypertensive_Disease_scaler.pkl")
except FileNotFoundError as e:
    raise RuntimeError(f"Model file not found: {e}")

class PredictionInput(BaseModel):
    education: int
    age: int
    BMI: float
    currentSmoker: int
    heartRate: int

@app.post("/predict")
def predict_hypertension(data: PredictionInput):
    """
    Predict hypertensive disease based on input features.
    
    - education: Education level (1-4)
    - age: Age in years
    - BMI: Body Mass Index
    - currentSmoker: Smoking status (0 or 1, or possibly 2?)
    - heartRate: Heart rate
    
    Returns:
        prediction: 0 (no hypertension) or 1 (hypertension)
    """
    # Convert input to array
    input_data = np.array([[data.education, data.age, data.BMI, data.currentSmoker, data.heartRate]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    return {"prediction": int(prediction[0])}

@app.get("/")
def read_root():
    return {"message": "Hypertensive Disease Prediction API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8070)