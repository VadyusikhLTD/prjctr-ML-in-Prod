from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from predictor import Predictor


class Payload(BaseModel):
    images: List[List[List[List[float]]]]


class Prediction(BaseModel):
    probs: List[List[float]]


app = FastAPI()
predictor = Predictor.load_from_model_registry()


@app.get("/health_check")
def health_check() -> int:
    return "ok"


@app.post("/predict", response_model=Prediction)
def predict(payload: Payload) -> Prediction:
    prediction = predictor.predict(in_img=torch.tensor(payload.images))
    return Prediction(probs=prediction.tolist())
