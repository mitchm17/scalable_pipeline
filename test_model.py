"""
Created on 2022-11-17 20:22:34
@author: Mitch Maletic

"""
import json
import numpy  as np
import pandas as pd
import pytest
from   fastapi.testclient                   import TestClient

from   scalable_pipeline.main               import app
from   scalable_pipeline.src.train_model    import TrainModel

client = TestClient(app)

@pytest.fixture
def data():
    data = pd.read_csv("./data/census.csv")
    return data
# end def

def test_categorical_column_values(data):
    categorical_columns = ['workclass', 'education', 'marital-status',
                           'occupation', 'relationship', 'race', 'sex',
                           'native-country', 'salary']

    for cat in categorical_columns:
        assert "?" not in data[cat].unique(), "There are still '?' entries in {}".format(cat)
    # end for
# end def

def test_stripped_columns(data):
    assert list(data.columns) == [x.strip() for x in data.columns], "Extra whitespace is still in the column names"
# end def

# Test the machine learning model that gets created
@pytest.fixture
def model():
    model = TrainModel("./data/census.csv")
    return model
# end def

def test_model_types(model):
    assert type(model.X_train) == np.ndarray
    assert type(model.X_test)  == np.ndarray
# end def

def test_model_scoring(model):
    assert type(model.precision)    == np.float64
    assert type(model.recall)       == np.float64
    assert type(model.fbeta)        == np.float64
# end def

# Test the FastAPI app
def test_root_response():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ['Hello: Reviewer! I hope you like my code!']
# end def

def test_app_bad_response():
    r = client.post("/make_model/")
    assert r.status_code != 200
# end def

def test_app_correct_response():
    data_path = {"file": "./data/census.csv"}
    r = client.post("/make_model/", data=json.dumps(data_path))
    assert r.status_code == 200
    assert 'model' in r.json() and 'precision' in r.json()
# end def

def test_app_low_salary_inference():
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
    response = client.post("/model_inference/", data=json.dumps(data))
    assert response.status_code == 200
    assert response.json()['model_inference'] == '<=50k'
# end def

def test_app_high_salary_inference():
    data = {
        "age": [56],
        "workclass": ["Self-emp-not-inc"],
        "fnlgt": [335605],
        "education": ["HS-grad"],
        "education-num": [9],
        "marital-status": ["Married-civ-spouse"],
        "occupation": ["Other-service"],
        "relationship": ["Husband"],
        "race": ["White"],
        "sex": ["Male"],
        "capital-gain": [0],
        "capital-loss": [1887],
        "hours-per-week": [50],
        "native-country": ["Canada"],
        }
    response = client.post("/model_inference/", data=json.dumps(data))
    assert response.status_code == 200
    assert response.json()['model_inference'] == '>50k'
# end def

def test_app_bad_inference_request():
    data = {
        "age": [39],
        "workclass": ["State-gov"],
        "fnlgt": [77516],
        "education": ["Bachelors"],
        "education-num": [13],
    }
    response = client.post("/model_inference", data=json.dumps(data))
    assert response.status_code != 200
# end def