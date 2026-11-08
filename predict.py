import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, create_model

with open("model.bin", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
features = artifact["features"]
threshold = artifact["threshold"]

# Input schema generated from the feature list saved by train.py:
# all 30 features are required and must be numeric
PatientFeatures = create_model(
    "PatientFeatures",
    **{name: (float, ...) for name in features},
)


class PredictResponse(BaseModel):
    malignant_probability: float
    diagnosis: int
    label: str
    threshold: float


app = FastAPI(title="Breast cancer prediction service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(patient: PatientFeatures):
    X = pd.DataFrame([patient.model_dump()])[features]
    proba = float(model.predict_proba(X)[0, 1])
    diagnosis = int(proba >= threshold)

    return PredictResponse(
        malignant_probability=round(proba, 4),
        diagnosis=diagnosis,
        label="malignant" if diagnosis else "benign",
        threshold=threshold,
    )