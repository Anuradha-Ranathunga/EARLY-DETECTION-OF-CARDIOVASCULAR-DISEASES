from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional
import os

# Set environment variable to fix pydantic compatibility issue
os.environ['PYDANTIC_PRIVATE_ALLOW_UNHANDLED_SCHEMA_TYPES'] = '1'

app = FastAPI(title="Myocardial Infarction Prediction API", description="API for predicting myocardial infarction complications using Gradient Boosting model")

# Load the model and scaler
try:
    model = joblib.load("Myocardial_Infarction_Predicter_Gradient_Boosting_Model.pkl")
    scaler = joblib.load("Scaler_Myocardial_Infarction.pkl")
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Define the input features based on the dataset
# Categorical columns that need encoding
categorical_cols = ['SEX', 'INF_ANAM', 'STENOK_AN', 'IBS_POST', 'SIM_GIPERT']

# All feature columns (excluding ID and target LET_IS)
feature_cols = [
    # AGE: int - Age of the patient
    'AGE',
    # SEX: int - Gender (0: Female, 1: Male)
    'SEX',
    # INF_ANAM: int - Infarction anamnesis (categorical)
    'INF_ANAM',
    # STENOK_AN: int - Stenocardia anamnesis (categorical)
    'STENOK_AN',
    # FK_STENOK: int - Functional class of stenocardia
    'FK_STENOK',
    # IBS_POST: int - Post-infarction angina (categorical)
    'IBS_POST',
    # IBS_NASL: int - Angina at rest
    'IBS_NASL',
    # GB: int - Gallbladder disease
    'GB',
    # SIM_GIPERT: int - Symptomatic hypertension (categorical)
    'SIM_GIPERT',
    # DLIT_AG: int - Duration of angina
    'DLIT_AG',
    # ZSN_A: int - Chronic heart failure
    'ZSN_A',
    # nr_11: int - ECG rhythm at admission (various codes)
    'nr_11',
    # nr_01: int - ECG rhythm at admission
    'nr_01',
    # nr_02: int - ECG rhythm at admission
    'nr_02',
    # nr_03: int - ECG rhythm at admission
    'nr_03',
    # nr_04: int - ECG rhythm at admission
    'nr_04',
    # nr_07: int - ECG rhythm at admission
    'nr_07',
    # nr_08: int - ECG rhythm at admission
    'nr_08',
    # np_01: int - ECG conduction at admission
    'np_01',
    # np_04: int - ECG conduction at admission
    'np_04',
    # np_05: int - ECG conduction at admission
    'np_05',
    # np_07: int - ECG conduction at admission
    'np_07',
    # np_08: int - ECG conduction at admission
    'np_08',
    # np_09: int - ECG conduction at admission
    'np_09',
    # np_10: int - ECG conduction at admission
    'np_10',
    # endocr_01: int - Endocrine system diseases
    'endocr_01',
    # endocr_02: int - Endocrine system diseases
    'endocr_02',
    # endocr_03: int - Endocrine system diseases
    'endocr_03',
    # zab_leg_01: int - Diabetes complications
    'zab_leg_01',
    # zab_leg_02: int - Diabetes complications
    'zab_leg_02',
    # zab_leg_03: int - Diabetes complications
    'zab_leg_03',
    # zab_leg_04: int - Diabetes complications
    'zab_leg_04',
    # zab_leg_06: int - Diabetes complications
    'zab_leg_06',
    # S_AD_KBRIG: float - Systolic blood pressure at admission
    'S_AD_KBRIG',
    # D_AD_KBRIG: float - Diastolic blood pressure at admission
    'D_AD_KBRIG',
    # S_AD_ORIT: float - Systolic blood pressure in ICU
    'S_AD_ORIT',
    # D_AD_ORIT: float - Diastolic blood pressure in ICU
    'D_AD_ORIT',
    # O_L_POST: int - Atrial fibrillation post-infarction
    'O_L_POST',
    # K_SH_POST: int - Cardiac shock post-infarction
    'K_SH_POST',
    # MP_TP_POST: int - Thrombolytic therapy post-infarction
    'MP_TP_POST',
    # SVT_POST: int - Supraventricular tachycardia post-infarction
    'SVT_POST',
    # GT_POST: int - Gastrointestinal tract complications
    'GT_POST',
    # FIB_G_POST: int - Fibrillation post-infarction
    'FIB_G_POST',
    # ant_im: int - Anterior wall infarction
    'ant_im',
    # lat_im: int - Lateral wall infarction
    'lat_im',
    # inf_im: int - Inferior wall infarction
    'inf_im',
    # post_im: int - Posterior wall infarction
    'post_im',
    # IM_PG_P: int - Infarction progression
    'IM_PG_P',
    # ritm_ecg_p_01: int - ECG rhythm post-hospitalization
    'ritm_ecg_p_01',
    # ritm_ecg_p_02: int - ECG rhythm post-hospitalization
    'ritm_ecg_p_02',
    # ritm_ecg_p_04: int - ECG rhythm post-hospitalization
    'ritm_ecg_p_04',
    # ritm_ecg_p_06: int - ECG rhythm post-hospitalization
    'ritm_ecg_p_06',
    # ritm_ecg_p_07: int - ECG rhythm post-hospitalization
    'ritm_ecg_p_07',
    # ritm_ecg_p_08: int - ECG rhythm post-hospitalization
    'ritm_ecg_p_08',
    # n_r_ecg_p_01: int - ECG rhythm normalization
    'n_r_ecg_p_01',
    # n_r_ecg_p_02: int - ECG rhythm normalization
    'n_r_ecg_p_02',
    # n_r_ecg_p_03: int - ECG rhythm normalization
    'n_r_ecg_p_03',
    # n_r_ecg_p_04: int - ECG rhythm normalization
    'n_r_ecg_p_04',
    # n_r_ecg_p_05: int - ECG rhythm normalization
    'n_r_ecg_p_05',
    # n_r_ecg_p_06: int - ECG rhythm normalization
    'n_r_ecg_p_06',
    # n_r_ecg_p_08: int - ECG rhythm normalization
    'n_r_ecg_p_08',
    # n_r_ecg_p_09: int - ECG rhythm normalization
    'n_r_ecg_p_09',
    # n_r_ecg_p_10: int - ECG rhythm normalization
    'n_r_ecg_p_10',
    # n_p_ecg_p_01: int - ECG conduction post-hospitalization
    'n_p_ecg_p_01',
    # n_p_ecg_p_03: int - ECG conduction post-hospitalization
    'n_p_ecg_p_03',
    # n_p_ecg_p_04: int - ECG conduction post-hospitalization
    'n_p_ecg_p_04',
    # n_p_ecg_p_05: int - ECG conduction post-hospitalization
    'n_p_ecg_p_05',
    # n_p_ecg_p_06: int - ECG conduction post-hospitalization
    'n_p_ecg_p_06',
    # n_p_ecg_p_07: int - ECG conduction post-hospitalization
    'n_p_ecg_p_07',
    # n_p_ecg_p_08: int - ECG conduction post-hospitalization
    'n_p_ecg_p_08',
    # n_p_ecg_p_09: int - ECG conduction post-hospitalization
    'n_p_ecg_p_09',
    # n_p_ecg_p_10: int - ECG conduction post-hospitalization
    'n_p_ecg_p_10',
    # n_p_ecg_p_11: int - ECG conduction post-hospitalization
    'n_p_ecg_p_11',
    # n_p_ecg_p_12: int - ECG conduction post-hospitalization
    'n_p_ecg_p_12',
    # fibr_ter_01: int - Fibrillation termination
    'fibr_ter_01',
    # fibr_ter_02: int - Fibrillation termination
    'fibr_ter_02',
    # fibr_ter_03: int - Fibrillation termination
    'fibr_ter_03',
    # fibr_ter_05: int - Fibrillation termination
    'fibr_ter_05',
    # fibr_ter_06: int - Fibrillation termination
    'fibr_ter_06',
    # fibr_ter_07: int - Fibrillation termination
    'fibr_ter_07',
    # fibr_ter_08: int - Fibrillation termination
    'fibr_ter_08',
    # GIPO_K: int - Hypokalemia
    'GIPO_K',
    # K_BLOOD: float - Potassium in blood
    'K_BLOOD',
    # GIPER_NA: int - Hypernatremia
    'GIPER_NA',
    # NA_BLOOD: float - Sodium in blood
    'NA_BLOOD',
    # ALT_BLOOD: float - Alanine aminotransferase
    'ALT_BLOOD',
    # AST_BLOOD: float - Aspartate aminotransferase
    'AST_BLOOD',
    # KFK_BLOOD: float - Creatine kinase
    'KFK_BLOOD',
    # L_BLOOD: float - Leukocytes
    'L_BLOOD',
    # ROE: float - Erythrocyte sedimentation rate
    'ROE',
    # TIME_B_S: int - Time from onset to hospitalization
    'TIME_B_S',
    # R_AB_1_n: int - Relapse of angina in hospital
    'R_AB_1_n',
    # R_AB_2_n: int - Relapse of angina in hospital
    'R_AB_2_n',
    # R_AB_3_n: int - Relapse of angina in hospital
    'R_AB_3_n',
    # NA_KB: int - Nitroglycerin in hospital
    'NA_KB',
    # NOT_NA_KB: int - No nitroglycerin in hospital
    'NOT_NA_KB',
    # LID_KB: int - Lidocaine in hospital
    'LID_KB',
    # NITR_S: int - Nitrates
    'NITR_S',
    # NA_R_1_n: int - Nitroglycerin at rest
    'NA_R_1_n',
    # NA_R_2_n: int - Nitroglycerin at rest
    'NA_R_2_n',
    # NA_R_3_n: int - Nitroglycerin at rest
    'NA_R_3_n',
    # NOT_NA_1_n: int - No nitroglycerin at rest
    'NOT_NA_1_n',
    # NOT_NA_2_n: int - No nitroglycerin at rest
    'NOT_NA_2_n',
    # NOT_NA_3_n: int - No nitroglycerin at rest
    'NOT_NA_3_n',
    # LID_S_n: int - Lidocaine
    'LID_S_n',
    # B_BLOK_S_n: int - Beta-blockers
    'B_BLOK_S_n',
    # ANT_CA_S_n: int - Calcium antagonists
    'ANT_CA_S_n',
    # GEPAR_S_n: int - Heparin
    'GEPAR_S_n',
    # ASP_S_n: int - Aspirin
    'ASP_S_n',
    # TIKL_S_n: int - Ticlopidine
    'TIKL_S_n',
    # TRENT_S_n: int - Ticlopidine
    'TRENT_S_n',
    # FIBR_PREDS: int - Fibrillation predisposition
    'FIBR_PREDS',
    # PREDS_TAH: int - Tachycardia predisposition
    'PREDS_TAH',
    # JELUD_TAH: int - Ventricular tachycardia
    'JELUD_TAH',
    # FIBR_JELUD: int - Ventricular fibrillation
    'FIBR_JELUD',
    # A_V_BLOK: int - Atrioventricular block
    'A_V_BLOK',
    # OTEK_LANC: int - Pulmonary edema
    'OTEK_LANC',
    # RAZRIV: int - Myocardial rupture
    'RAZRIV',
    # DRESSLER: int - Dressler's syndrome
    'DRESSLER',
    # ZSN: int - Chronic heart failure
    'ZSN',
    # REC_IM: int - Recurrent infarction
    'REC_IM',
    # P_IM_STEN: int - Post-infarction stenocardia
    'P_IM_STEN'
]

class PredictionInput(BaseModel):
    # Explicit fields for all model inputs. Use Optional[...] to allow missing fields.
    AGE: Optional[int] = None
    SEX: Optional[int] = None
    INF_ANAM: Optional[int] = None
    STENOK_AN: Optional[int] = None
    FK_STENOK: Optional[int] = None
    IBS_POST: Optional[int] = None
    IBS_NASL: Optional[int] = None
    GB: Optional[int] = None
    SIM_GIPERT: Optional[int] = None
    DLIT_AG: Optional[int] = None
    ZSN_A: Optional[int] = None
    nr_11: Optional[int] = None
    nr_01: Optional[int] = None
    nr_02: Optional[int] = None
    nr_03: Optional[int] = None
    nr_04: Optional[int] = None
    nr_07: Optional[int] = None
    nr_08: Optional[int] = None
    np_01: Optional[int] = None
    np_04: Optional[int] = None
    np_05: Optional[int] = None
    np_07: Optional[int] = None
    np_08: Optional[int] = None
    np_09: Optional[int] = None
    np_10: Optional[int] = None
    endocr_01: Optional[int] = None
    endocr_02: Optional[int] = None
    endocr_03: Optional[int] = None
    zab_leg_01: Optional[int] = None
    zab_leg_02: Optional[int] = None
    zab_leg_03: Optional[int] = None
    zab_leg_04: Optional[int] = None
    zab_leg_06: Optional[int] = None
    S_AD_KBRIG: Optional[float] = None
    D_AD_KBRIG: Optional[float] = None
    S_AD_ORIT: Optional[float] = None
    D_AD_ORIT: Optional[float] = None
    O_L_POST: Optional[int] = None
    K_SH_POST: Optional[int] = None
    MP_TP_POST: Optional[int] = None
    SVT_POST: Optional[int] = None
    GT_POST: Optional[int] = None
    FIB_G_POST: Optional[int] = None
    ant_im: Optional[int] = None
    lat_im: Optional[int] = None
    inf_im: Optional[int] = None
    post_im: Optional[int] = None
    IM_PG_P: Optional[int] = None
    ritm_ecg_p_01: Optional[int] = None
    ritm_ecg_p_02: Optional[int] = None
    ritm_ecg_p_04: Optional[int] = None
    ritm_ecg_p_06: Optional[int] = None
    ritm_ecg_p_07: Optional[int] = None
    ritm_ecg_p_08: Optional[int] = None
    n_r_ecg_p_01: Optional[int] = None
    n_r_ecg_p_02: Optional[int] = None
    n_r_ecg_p_03: Optional[int] = None
    n_r_ecg_p_04: Optional[int] = None
    n_r_ecg_p_05: Optional[int] = None
    n_r_ecg_p_06: Optional[int] = None
    n_r_ecg_p_08: Optional[int] = None
    n_r_ecg_p_09: Optional[int] = None
    n_r_ecg_p_10: Optional[int] = None
    n_p_ecg_p_01: Optional[int] = None
    n_p_ecg_p_03: Optional[int] = None
    n_p_ecg_p_04: Optional[int] = None
    n_p_ecg_p_05: Optional[int] = None
    n_p_ecg_p_06: Optional[int] = None
    n_p_ecg_p_07: Optional[int] = None
    n_p_ecg_p_08: Optional[int] = None
    n_p_ecg_p_09: Optional[int] = None
    n_p_ecg_p_10: Optional[int] = None
    n_p_ecg_p_11: Optional[int] = None
    n_p_ecg_p_12: Optional[int] = None
    fibr_ter_01: Optional[int] = None
    fibr_ter_02: Optional[int] = None
    fibr_ter_03: Optional[int] = None
    fibr_ter_05: Optional[int] = None
    fibr_ter_06: Optional[int] = None
    fibr_ter_07: Optional[int] = None
    fibr_ter_08: Optional[int] = None
    GIPO_K: Optional[int] = None
    K_BLOOD: Optional[float] = None
    GIPER_NA: Optional[int] = None
    NA_BLOOD: Optional[float] = None
    ALT_BLOOD: Optional[float] = None
    AST_BLOOD: Optional[float] = None
    KFK_BLOOD: Optional[float] = None
    L_BLOOD: Optional[float] = None
    ROE: Optional[float] = None
    TIME_B_S: Optional[int] = None
    R_AB_1_n: Optional[int] = None
    R_AB_2_n: Optional[int] = None
    R_AB_3_n: Optional[int] = None
    NA_KB: Optional[int] = None
    NOT_NA_KB: Optional[int] = None
    LID_KB: Optional[int] = None
    NITR_S: Optional[int] = None
    NA_R_1_n: Optional[int] = None
    NA_R_2_n: Optional[int] = None
    NA_R_3_n: Optional[int] = None
    NOT_NA_1_n: Optional[int] = None
    NOT_NA_2_n: Optional[int] = None
    NOT_NA_3_n: Optional[int] = None
    LID_S_n: Optional[int] = None
    B_BLOK_S_n: Optional[int] = None
    ANT_CA_S_n: Optional[int] = None
    GEPAR_S_n: Optional[int] = None
    ASP_S_n: Optional[int] = None
    TIKL_S_n: Optional[int] = None
    TRENT_S_n: Optional[int] = None
    FIBR_PREDS: Optional[int] = None
    PREDS_TAH: Optional[int] = None
    JELUD_TAH: Optional[int] = None
    FIBR_JELUD: Optional[int] = None
    A_V_BLOK: Optional[int] = None
    OTEK_LANC: Optional[int] = None
    RAZRIV: Optional[int] = None
    DRESSLER: Optional[int] = None
    ZSN: Optional[int] = None
    REC_IM: Optional[int] = None
    P_IM_STEN: Optional[int] = None
    
    def clean_dict(self) -> Dict[str, Any]:
        """Return a dict of input values with None values removed and basic type casting where appropriate."""
        cleaned: Dict[str, Any] = {}
        for k, v in self.dict().items():
            if v is None:
                continue
            # Cast floats/ints that may have come in as strings
            if k in {"AGE", "SEX", "INF_ANAM", "STENOK_AN", "FK_STENOK"}:
                try:
                    cleaned[k] = int(v)
                except Exception:
                    cleaned[k] = v
            elif k in {"K_BLOOD", "NA_BLOOD", "ALT_BLOOD", "AST_BLOOD", "KFK_BLOOD", "L_BLOOD", "ROE"}:
                try:
                    cleaned[k] = float(v)
                except Exception:
                    cleaned[k] = v
            else:
                cleaned[k] = v
        return cleaned

    @validator('AGE')
    def age_must_be_reasonable(cls, v):
        if v is None:
            return v
        if v < 0 or v > 120:
            raise ValueError('AGE must be between 0 and 120')
        return v

    @validator('K_BLOOD', 'NA_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD', 'L_BLOOD', 'ROE', pre=True)
    def floats_must_be_non_negative(cls, v):
        if v is None:
            return v
        try:
            val = float(v)
        except Exception:
            raise ValueError('must be a number')
        if val < 0:
            raise ValueError('must be non-negative')
        return val

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "AGE": 65,
                "SEX": 1,
                "K_BLOOD": 4.2,
                "NA_BLOOD": 140.0,
                "INF_ANAM": 0
            }
        }

def preprocess_input(data: Dict[str, Any]) -> np.ndarray:
    """Preprocess the input data to match the training data format"""
    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Handle missing values for numeric columns
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Handle missing values for categorical columns
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
            df[col] = df[col].astype('category')

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ensure all expected columns are present (add missing columns with 0)
    if hasattr(scaler, 'feature_names_in_'):
        expected_cols = list(scaler.feature_names_in_)
    else:
        expected_cols = list(df_encoded.columns)
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns to match training data
    df_encoded = df_encoded[expected_cols]

    # Scale the features
    scaled_features = scaler.transform(df_encoded)

    return scaled_features

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")

    try:
        # Convert Pydantic model to dict and drop None values
        input_dict = {k: v for k, v in input_data.dict().items()}

        # Preprocess the input
        processed_data = preprocess_input(input_dict)

        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)

        # Get the predicted class and probabilities
        predicted_class = int(prediction[0])
        probabilities = {f"class_{i}": float(prob) for i, prob in enumerate(prediction_proba[0])}

        return {
            "prediction": predicted_class,
            "probabilities": probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Myocardial Infarction Prediction API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
