
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/final/final_xgboost_model.joblib"

THRESHOLD = 0.5

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


# --------------------------------------------------
# 2. LOAD THE FINAL MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Final model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="CreditGuard API",
    description=(
        "Credit card fraud detection API "
        "powered by the trained XGBoost model."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# 4. REQUEST VALIDATION
# --------------------------------------------------

class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        description="All 30 transaction features."
    )

    @model_validator(mode="after")
    def validate_features(self):
        expected = set(FEATURE_COLUMNS)
        supplied = set(self.features.keys())

        missing = expected - supplied
        extra = supplied - expected

        if missing:
            raise ValueError(
                f"Missing features: {sorted(missing)}"
            )

        if extra:
            raise ValueError(
                f"Unexpected features: {sorted(extra)}"
            )

        for name, value in self.features.items():
            if not pd.notna(value):
                raise ValueError(
                    f"{name} must be a finite number."
                )

        if self.features["Time"] < 0:
            raise ValueError("Time cannot be negative.")

        if self.features["Amount"] < 0:
            raise ValueError("Amount cannot be negative.")

        return self


# --------------------------------------------------
# 5. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to CreditGuard API",
        "status": "running",
        "model": "Optimised XGBoost",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


# --------------------------------------------------
# 6. FRAUD PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_transaction(transaction: TransactionRequest):

    try:
        # Preserve the exact feature order used in training.
        input_data = pd.DataFrame(
            [
                [
                    transaction.features[column]
                    for column in FEATURE_COLUMNS
                ]
            ],
            columns=FEATURE_COLUMNS,
        )

        # Get predicted fraud probability.
        fraud_probability = float(
            model.predict_proba(input_data)[0][1]
        )

        # Apply the previously selected threshold.
        is_fraud = fraud_probability >= THRESHOLD

        prediction = (
            "FRAUD" if is_fraud else "LEGITIMATE"
        )

        risk_category = (
            "HIGH" if is_fraud else "LOW"
        )

        return {
            "prediction": prediction,
            "fraud_probability": round(
                fraud_probability, 6
            ),
            "fraud_probability_percentage": round(
                fraud_probability * 100, 2
            ),
            "risk_category": risk_category,
            "threshold": THRESHOLD,
            "model": "Optimised XGBoost",
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )
