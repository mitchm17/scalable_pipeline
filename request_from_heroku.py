"""
Created on 2022-11-20 11:00:42
@author: Mitch Maletic

"""

import json
import requests

data = {"file": "./data/census.csv"}
url  = "https://maletic-scalable-pipeline-app.herokuapp.com/model_inference/"
response = requests.post(url, data=json.dumps(data))
print("Response status code: {}".format(response.status_code))
print("Response result: {}".format(response.json()))