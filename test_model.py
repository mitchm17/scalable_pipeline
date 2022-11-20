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
    assert r.json() == ['Hello: Reviewer! Hope you like my code!']
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