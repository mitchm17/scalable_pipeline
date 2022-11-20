"""
Created on 2022-11-17 22:40:09
@author: Mitch Maletic

"""

import argparse
import json
import os
from fastapi.testclient import TestClient
from main               import app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Testing file to send requests to ML training app")

    parser.add_argument("-p", "--path", type=str,
                        help="Path to data to be used")

    args        = parser.parse_args()
    if args.path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "census.csv")
    else:
        path = args.path
    # end if
    data        = {"file": path}
    client      = TestClient(app)
    r           = client.get("/")
    print(r.json())
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
    response    = client.post("/model_inference/", data=json.dumps(data))
    print(response.json())
# end if