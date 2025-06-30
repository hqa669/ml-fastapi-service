# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:08:27 2025

@author: hqa66
"""

import requests

def test_predict_endpoint():
    input_data = {
        "Pclass": 2, "Sex": "female", "Age": 25, "Fare": 15.5, "Embarked": "C"
    }
    res = requests.post("http://localhost:8000/predict", json=input_data)
    assert res.status_code == 200
    result = res.json()
    assert "prediction" in result
    assert "probability" in result