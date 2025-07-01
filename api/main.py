# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:20:15 2025

@author: hqa66
"""

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import List, Union, Literal
import joblib
import pandas as pd
import logging
import json
from datetime import datetime

# Load trained model
model = joblib.load("model/final_model_xgb.joblib")

# Configure logging
logging.basicConfig(
    filename="logs/prediction_log.jsonl",
    level=logging.INFO,
    format="%(message)s"
)

# Define input schema
class InputData(BaseModel):
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: float = Field(..., gt=0, le=100)
    Fare: float = Field(..., ge=0)
    Embarked: Literal["C", "Q", "S"]

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "Service online."}

@app.post("/predict")
async def predict(request: Request, data: Union[InputData, List[InputData]]):
    # Normalize input
    records = [data.dict()] if isinstance(data, InputData) else [d.dict() for d in data]
    df = pd.DataFrame(records)

    # Predict
    preds = model.predict(df).tolist()
    probs = model.predict_proba(df)[:, 1].tolist()

    # Log predictions
    for record, pred, prob in zip(records, preds, probs):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_ip": request.client.host,
            "request": record,
            "prediction": int(pred),
            "probability": round(prob, 4)
        }
        logging.info(json.dumps(log_entry))

    # Build response
    return [
        {"prediction": int(pred), "probability": round(prob, 4)}
        for pred, prob in zip(preds, probs)
    ]