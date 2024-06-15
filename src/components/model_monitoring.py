import pandas as pd 
import numpy as np 
import sys
import os
import mlflow
import mlflow.sklearn
import pickle
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


class ModelEvaluation:
    def __init__(self):
        logging.info("Model Evaluation Started")

    def eval_metrics(self,actual, pred):

        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])

            model_path=os.path.join("artifacts","model.pkl")
             
            model=load_object(model_path)
            # Set the model registry to use the default local filesystem

            mlflow.set_registry_uri("")


            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

            print(tracking_url_type_store)

            with mlflow.start_run():
                prediction=model.predict(X_test)
                (rmse,mae,r2)=self.eval_metrics(y_test,prediction)
                alpha=0.8


                mlflow.log_param("alpha", alpha)
                

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise CustomException(e,sys)

