"""
Created on 2022-11-20 11:00:42
@author: Mitch Maletic

"""

import json
import requests

data = {
        "age": [39],
        "workclass": ["State-gov"],
        "fnlgt": [77516],
        "education": ["Bachelors"],
        "education-num": [13],
        "marital-status": ["Never-married"],
        "occupation": ["Adm-clerical"],
        "relationship": ["Not-in-family"],
        "race": ["White"],
        "sex": ["Male"],
        "capital-gain": [2174],
        "capital-loss": [0],
        "hours-per-week": [40],
        "native-country": ["United-States"],
        }
url  = "https://maletic-scalable-pipeline-app.herokuapp.com/model_inference/"
response = requests.post(url, data=json.dumps(data))
print("Response status code: {}".format(response.status_code))
print("Response result: {}".format(response.json()))