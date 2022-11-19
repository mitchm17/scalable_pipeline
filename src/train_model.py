"""
Created on 2022-11-17 22:50:17
@author: Mitch Maletic

"""

import numpy as np
import os
import pandas as pd
import pickle

from sklearn.model_selection    import train_test_split
from sklearn.metrics            import (
                                        fbeta_score,
                                        precision_score,
                                        recall_score
                                        )
from sklearn.ensemble           import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder

class TrainModel():

    def __init__(self, data_path):
        self.data_path = data_path

        self.cat_features = [
                             "workclass",
                             "education",
                             "marital-status",
                             "occupation",
                             "relationship",
                             "race",
                             "sex",
                             "native-country",
                            ]
        self._load_data()
        self._get_splits()
        self._process_training_data()
        self._process_testing_data()
        self._train_model()
        self._model_predictions()
        self._score_model()
        self._save_model()
        self._slice_data()
    # end def

    def _load_data(self):
        self.data    = pd.read_csv(self.data_path)
    # end def

    def _get_splits(self):
        self.train, self.test = train_test_split(self.data,
                                                 test_size=0.20,
                                                 random_state=17)
    # end def

    def _process_training_data(self):
        X_train, y_train, encoder, lb   = self._process_data(
                                                       self.train,
                                                       categorical_features=self.cat_features,
                                                       label="salary",
                                                       training=True
                                                      )
        self.X_train, self.y_train      = X_train, y_train
        self.training_encoder           = encoder
        self.lb                         = lb
    # end def

    def _process_testing_data(self):
        X_test, y_test, _, _        = self._process_data(
                                                   self.test,
                                                   categorical_features=self.cat_features,
                                                   label="salary",
                                                   training=False,
                                                   encoder=self.training_encoder,
                                                   lb=self.lb
                                                  )
        self.X_test, self.y_test    = X_test, y_test
    # end def

    def _train_model(self):
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
        self.model = RandomForestClassifier(random_state=random_seed)
        self.model.fit(self.X_train, self.y_train)
    # end def

    def _model_predictions(self):
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
        self.preds = self.model.predict(self.X_test)
    # end def

    def _score_model(self, y=None, preds=None):
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
        if y is None and preds is None:
            self.fbeta       = fbeta_score(self.y_test, self.preds, beta=1,
                                        zero_division=1)
            self.precision   = precision_score(self.y_test, self.preds,
                                            zero_division=1)
            self.recall      = recall_score(self.y_test, self.preds,
                                            zero_division=1)
        else:
            fbeta            = fbeta_score(y, preds, beta=1, zero_division=1)
            precision        = precision_score(y, preds, zero_division=1)
            recall           = recall_score(y, preds, zero_division=1)
            return (fbeta, precision, recall)
        # end if
    # end def

    def _save_model(self):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "model", "model.pkl")
        with open(model_path, 'wb') as save_model:
            pickle.dump(self.model, save_model)
        # end with

        self.encoder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "model", "encoder.pkl")
        with open(self.encoder_path, "wb") as save_encoder:
            pickle.dump(self.training_encoder, save_encoder)
        # end with
    # end def

    def _process_data(self, X, categorical_features=[], label=None,
                      training=True, encoder=None, lb=None):
        """ Process the data used in the machine learning pipeline.

        Processes the data using one hot encoding for the categorical features and a
        label binarizer for the labels. This can be used in either training or
        inference/validation.

        Note: depending on the type of model used, you may want to add in functionality that
        scales the continuous data.

        Inputs
        ------
        X : pd.DataFrame
            Dataframe containing the features and label. Columns in `categorical_features`
        categorical_features: list[str]
            List containing the names of the categorical features (default=[])
        label : str
            Name of the label column in `X`. If None, then an empty array will be returned
            for y (default=None)
        training : bool
            Indicator if training mode or inference/validation mode.
        encoder : sklearn.preprocessing._encoders.OneHotEncoder
            Trained sklearn OneHotEncoder, only used if training=False.
        lb : sklearn.preprocessing._label.LabelBinarizer
            Trained sklearn LabelBinarizer, only used if training=False.

        Returns
        -------
        X : np.array
            Processed data.
        y : np.array
            Processed labels if labeled=True, otherwise empty np.array.
        encoder : sklearn.preprocessing._encoders.OneHotEncoder
            Trained OneHotEncoder if training is True, otherwise returns the encoder passed
            in.
        lb : sklearn.preprocessing._label.LabelBinarizer
            Trained LabelBinarizer if training is True, otherwise returns the binarizer
            passed in.
        """

        if label is not None:
            y = X[label]
            X = X.drop([label], axis=1)
        else:
            y = np.array([])

        X_categorical   = X[categorical_features].values
        X_continuous    = X.drop(*[categorical_features], axis=1)

        if training is True:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            lb = LabelBinarizer()
            X_categorical = encoder.fit_transform(X_categorical)
            y = lb.fit_transform(y.values).ravel()
        else:
            X_categorical = encoder.transform(X_categorical)
            try:
                y = lb.transform(y.values).ravel()
            # Catch the case where y is None because we're doing inference.
            except AttributeError:
                pass

        X = np.concatenate([X_continuous, X_categorical], axis=1)
        return X, y, encoder, lb
    # end def

    def _slice_data(self):
        """ Function for calculating descriptive stats on slices of the dataset.
        Referencing the knowledge question: https://knowledge.udacity.com/questions/909273
        """
        _, test_set  = train_test_split(self.data, test_size=0.20,
                                        random_state=17,
                                        stratify=self.data.salary)
        encoder      = pd.read_pickle(self.encoder_path)
        ouptut_lines = []
        for cat_feat in self.cat_features:
            for cls in test_set[cat_feat].unique():
                df_temp         = test_set[test_set[cat_feat] == cls]
                X_test, _, _, _ = self._process_data(df_temp, self.cat_features,
                                                     encoder=encoder,
                                                     label='salary', lb=self.lb,
                                                     training=False)
                y_preds         = self.model.predict(X_test)
                y               = df_temp.iloc[:,-1:]
                lb              = LabelEncoder()
                y               = lb.fit_transform(np.ravel(y))
                fbeta, prec, recall = self._score_model(y, y_preds)
                feature_output  = f"\n{cat_feat.upper()} Stats for {cls} Class: "
                feature_output += f"{cat_feat} precision: {prec:.4f}, "
                feature_output += f"{cat_feat} recall: {recall:.4f}, "
                feature_output += f"{cat_feat} fbeta: {fbeta:.4f}"
                ouptut_lines.append(feature_output)
            # end for
        # end for cat_feat

        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "slice_output.txt")
        with open(output_path, 'w') as output:
            output.writelines(ouptut_lines)
        # end with
    # end def
# end class

if __name__ == "__main__":
    tm = TrainModel("/home/mitch/scalable_pipeline/data/census.csv")