import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.utils import save_object
from dataclasses import dataclass
from pathlib import Path
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_datatransformation(self):
        logging.info("Data trnsformation Process of creation preprocessor started")
        try:
            categorical_columns=['cut', 'color', 'clarity']
            numerical_columns=['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            num_pipeline=Pipeline (
            steps=[
             ("imputer",SimpleImputer()),
            ("scaler",StandardScaler())
             ]
            )

            cat_pipeline=Pipeline(

            steps=[
             ("imputer",SimpleImputer(strategy="most_frequent")),
             ("ordinalencoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))

              ]
              )
            
            preprocessor=ColumnTransformer(
               [
               ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
               ]
             )
            
            return preprocessor

        except Exception as e:
            logging.info("There is error in Get Data Transformation Preprocessing object crteation")

    def initiate_datatransformation(self,train_path,test_path):
        logging.info("Initiating data transformation process")
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Reading Train and Test data Completed")

            preprocessing_obj=self.get_datatransformation()

            target_column='price'
            drop_columns = [target_column,'id']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1) #xtrain
            target_feature_train_df=train_df[target_column] #ytrain
            
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1) #xtest
            target_feature_test_df=test_df[target_column]   #ytest


            input_feature_train_df_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_df_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")

            return (
                train_arr,
                test_arr
            )




        except Exception as e:
            raise CustomException(e,sys)
        

# if __name__=="__main__":
#     a=DataIngestion()
#     train_path,test_path=a.initiate_dataingestion()
#     b=DataTransformation()
#     b.initiate_datatransformation(train_path,test_path)
