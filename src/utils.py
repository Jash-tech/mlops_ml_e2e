import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)


    except Exception as e:
        logging.info('Exception Occured in save_object function utils')
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            r2score = r2_score(y_test,y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae = mean_absolute_error(y_test, y_test_pred)

            report['R2Score'] =  r2score
            report['RMSE'] =  rmse
            report['MAE'] =  mae

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)


