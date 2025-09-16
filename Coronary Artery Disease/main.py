from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn
from typing import Union

app = FastAPI(title="Coronary Artery Disease Prediction API", version="1.0.0")

# Load the model and scaler
model = joblib.load("CAD_Predictor_RF_BestModel.pkl")
scaler = joblib.load("scaler_CAD.pkl")

# Feature names (from the dataset, excluding 'Cath')
feature_names = [
    'Age', 'Weight', 'Length', 'Sex', 'BMI', 'DM', 'HTN', 'Current Smoker', 'EX-Smoker', 'FH', 'Obesity', 'CRF', 'CVA', 'Airway disease', 'Thyroid Disease', 'CHF', 'DLP', 'BP', 'PR', 'Edema', 'Weak Peripheral Pulse', 'Lung rales', 'Systolic Murmur', 'Diastolic Murmur', 'Typical Chest Pain', 'Dyspnea', 'Function Class', 'Atypical', 'Nonanginal', 'Exertional CP', 'LowTH Ang', 'Q Wave', 'St Elevation', 'St Depression', 'Tinversion', 'LVH', 'Poor R Progression', 'BBB', 'FBS', 'CR', 'TG', 'LDL', 'HDL', 'BUN', 'ESR', 'HB', 'K', 'Na', 'WBC', 'Lymph', 'Neut', 'PLT', 'EF-TTE', 'Region RWMA', 'VHD'
]

# Encoding mappings
sex_mapping = {'Fmale': 0, 'Male': 1}
binary_mapping = {'N': 0, 'Y': 1}
vhd_mapping = {'N': 0, 'mild': 1, 'Moderate': 2, 'Severe': 3}

class PredictionInput(BaseModel):
    Age: float
    Weight: float
    Length: float
    Sex: str  # 'Male' or 'Fmale'
    BMI: float
    DM: str  # 'Y' or 'N'
    HTN: str
    Current_Smoker: str
    EX_Smoker: str
    FH: str
    Obesity: str
    CRF: str
    CVA: str
    Airway_disease: str
    Thyroid_Disease: str
    CHF: str
    DLP: str
    BP: float
    PR: float
    Edema: str
    Weak_Peripheral_Pulse: str
    Lung_rales: str
    Systolic_Murmur: str
    Diastolic_Murmur: str
    Typical_Chest_Pain: str
    Dyspnea: str
    Function_Class: int  # 0-3
    Atypical: str
    Nonanginal: str
    Exertional_CP: str
    LowTH_Ang: str
    Q_Wave: str
    St_Elevation: str
    St_Depression: str
    Tinversion: str
    LVH: str
    Poor_R_Progression: str
    BBB: str
    FBS: float
    CR: float
    TG: float
    LDL: float
    HDL: float
    BUN: float
    ESR: float
    HB: float
    K: float
    Na: float
    WBC: float
    Lymph: float
    Neut: float
    PLT: float
    EF_TTE: float
    Region_RWMA: int  # 0-4
    VHD: str  # 'N', 'mild', 'Moderate', 'Severe'

def encode_input(data: PredictionInput) -> np.ndarray:
    # Encode categorical variables
    encoded = []
    encoded.append(data.Age)
    encoded.append(data.Weight)
    encoded.append(data.Length)
    encoded.append(sex_mapping[data.Sex])
    encoded.append(data.BMI)
    encoded.append(binary_mapping[data.DM])
    encoded.append(binary_mapping[data.HTN])
    encoded.append(binary_mapping[data.Current_Smoker])
    encoded.append(binary_mapping[data.EX_Smoker])
    encoded.append(binary_mapping[data.FH])
    encoded.append(binary_mapping[data.Obesity])
    encoded.append(binary_mapping[data.CRF])
    encoded.append(binary_mapping[data.CVA])
    encoded.append(binary_mapping[data.Airway_disease])
    encoded.append(binary_mapping[data.Thyroid_Disease])
    encoded.append(binary_mapping[data.CHF])
    encoded.append(binary_mapping[data.DLP])
    encoded.append(data.BP)
    encoded.append(data.PR)
    encoded.append(binary_mapping[data.Edema])
    encoded.append(binary_mapping[data.Weak_Peripheral_Pulse])
    encoded.append(binary_mapping[data.Lung_rales])
    encoded.append(binary_mapping[data.Systolic_Murmur])
    encoded.append(binary_mapping[data.Diastolic_Murmur])
    encoded.append(binary_mapping[data.Typical_Chest_Pain])
    encoded.append(binary_mapping[data.Dyspnea])
    encoded.append(data.Function_Class)
    encoded.append(binary_mapping[data.Atypical])
    encoded.append(binary_mapping[data.Nonanginal])
    encoded.append(binary_mapping[data.Exertional_CP])
    encoded.append(binary_mapping[data.LowTH_Ang])
    encoded.append(binary_mapping[data.Q_Wave])
    encoded.append(binary_mapping[data.St_Elevation])
    encoded.append(binary_mapping[data.St_Depression])
    encoded.append(binary_mapping[data.Tinversion])
    encoded.append(binary_mapping[data.LVH])
    encoded.append(binary_mapping[data.Poor_R_Progression])
    encoded.append(binary_mapping[data.BBB])
    encoded.append(data.FBS)
    encoded.append(data.CR)
    encoded.append(data.TG)
    encoded.append(data.LDL)
    encoded.append(data.HDL)
    encoded.append(data.BUN)
    encoded.append(data.ESR)
    encoded.append(data.HB)
    encoded.append(data.K)
    encoded.append(data.Na)
    encoded.append(data.WBC)
    encoded.append(data.Lymph)
    encoded.append(data.Neut)
    encoded.append(data.PLT)
    encoded.append(data.EF_TTE)
    encoded.append(data.Region_RWMA)
    encoded.append(vhd_mapping[data.VHD])

    return np.array(encoded).reshape(1, -1)

@app.post("/predict")
def predict_cad(input_data: PredictionInput):
    # Encode and scale the input
    encoded_input = encode_input(input_data)
    scaled_input = scaler.transform(encoded_input)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]  # Probability of CAD (class 1)

    result = {
        "prediction": int(prediction),
        "probability": float(probability),
        "diagnosis": "CAD" if prediction == 1 else "No CAD"
    }

    return result

@app.get("/")
def root():
    return {"message": "Coronary Artery Disease Prediction API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
