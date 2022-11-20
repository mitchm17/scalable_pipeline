"""
Created on 2022-11-20 14:54:46
@author: Mitch Maletic

"""

import numpy as np
import os
import pandas as pd
import pickle

def load_models():
    parent_dir      = os.path.dirname(os.path.dirname(__file__))
    model_path      = os.path.join(parent_dir, "model", "model.pkl")
    encoder_path    = os.path.join(parent_dir, "model", "encoder.pkl")

    with open(model_path, "rb") as model_pkl:
        model = pickle.load(model_pkl)
    # end with

    with open(encoder_path, "rb") as encoder_pkl:
        encoder = pickle.load(encoder_pkl)
    # end with
    return model, encoder
# end def

def process_data(data, encoder):
    cat_features = [
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",
                    ]
    X_categorical = data[cat_features].values
    X_continuous  = data.drop(*[cat_features], axis=1)

    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X

def model_inference(data_dict):
    data            = pd.DataFrame(data_dict.dict())
    data.columns    = data.columns.str.replace("_", "-")
    model, encoder  = load_models()
    X               = process_data(data, encoder)
    preds           = model.predict(X)[0]
    return preds
# end def