import hopsworks
import os
import numpy as np
import hsfs
import joblib

class Predict(object):

    def __init__(self):
        """ Initializes the serving state, reads a trained model"""
        # get feature store handle
        fs_conn = hsfs.connection()
        self.fs = fs_conn.get_feature_store()

        # get feature view
        self.fv = self.fs.get_feature_view("loan_approval_feature_view", 1)

        # initialize serving
        self.fv.init_serving(0.5)

        # load the trained model
        self.model = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/logistic_regression_model.pkl")
        print("Initialization Complete")

    def predict(self, inputs):
        """ Serves a prediction request usign a trained model"""
        return self.model.predict(np.asarray(inputs).reshape(1, -1)).tolist()