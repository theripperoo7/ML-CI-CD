from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.joblib")

class InputData(BaseModel):
    input: list[float]

@app.get("/")
def read_root():
    return {"message": "Witaj w API modelu regresji liniowej!"}

@app.post("/predict")
def predict(data: InputData):
    expected_features = 2 
    if len(data.input) != expected_features:
        return {"error": f"Niewłaściwa liczba cech. Oczekiwano {expected_features}, otrzymano {len(data.input)}"}
    prediction = model.predict([data.input])
    return {"prediction": float(prediction[0])}

@app.get("/info")
def info():
    return {
        "model_type": type(model).__name__,
        "num_features": model.coef_.shape[0] if hasattr(model, "coef_") else "unknown"
    }

@app.get("/health")
def health():
    return {"status": "ok"}
