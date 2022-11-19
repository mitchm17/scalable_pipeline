"""
Created on 2022-11-17 22:40:09
@author: Mitch Maletic

"""

import argparse
import json
from fastapi.testclient import TestClient
from main               import app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Testing file to send requests to ML training app")

    parser.add_argument("-p", "--path", type=str,
                        help="Path to data to be used")

    args        = parser.parse_args()
    data        = {"file": args.path}
    client      = TestClient(app)
    r           = client.get("/")
    print(r.json())
    response    = client.post("/model_inference/", data=json.dumps(data))
    print(response.json())
# end if