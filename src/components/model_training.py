import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from dataclasses import dataclass
from pathlib import Path
from src.utils import save_object,evaluate_model


@dataclass 
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:

            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                 )

            models={
            'Elasticnet':ElasticNet()
             }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            
            
            
            best_model = models['Elasticnet']


            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

        except Exception as e:
            raise CustomException(e,sys)