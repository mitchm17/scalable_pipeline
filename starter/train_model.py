# Script to train machine learning model.
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data    import process_data
from ml.model   import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger(__name__)
for key in (logging.Logger.manager.loggerDict):
    # if you have more than one logger you want to run,
    # change line below to bey key in [list of logs]
    if key != __name__:
        logging.getLogger(key).disabled = True
    # end if
# end for

# Add code to load in the data.
logger.info("Reading in data")
data    = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "data", "census.csv")
                      )

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Splitting dataset into training and testing splits")
train, test = train_test_split(data, test_size=0.20)

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
logger.info("Processing training data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
logger.info("Processing testing data")
X_test, y_test, test_encoder, test_lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
logger.info("Training and scoring the model")
model = train_model(X_train, y_train)
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logger.info("Model: {}, Precision: {}, Recall: {}, Fbeta: {}".format(
    model, precision, recall, fbeta
))
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "model", "model.pkl")
logger.info("Saving model to {}".format(model_path))
with open(model_path, 'wb') as save_model:
    pickle.dump(model, save_model)
# end with