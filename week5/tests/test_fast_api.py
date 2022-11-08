import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np

from fast_api_serving import app

client = TestClient(app)


@pytest
def test_health_check():
    response = client.get("/health_check")
    assert response.status_code == 200
    assert response.json() == "ok"


@pytest
def test_predict():
    a = np.zeros((1, 1, 28, 28)).tolist()
    response = client.post("/predict", json={"images": a})
    assert response.status_code == 200
    assert len(response.json()["probs"][0]) == 10
