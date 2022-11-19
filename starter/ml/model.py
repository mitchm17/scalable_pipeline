"""
Created on 2022-11-14 20:33:55
@author: Mitch Maletic

"""


from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    random_seed = 17
    model = RandomForestClassifier(random_state=random_seed)
    model.fit(X_train, y_train)

    return model
# end def


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta       = fbeta_score(y, preds, beta=1, zero_division=1)
    precision   = precision_score(y, preds, zero_division=1)
    recall      = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta
# end def

def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.linear_model.LinearRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds
# end def