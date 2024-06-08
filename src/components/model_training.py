import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split

@dataclass
class ModelTrainingConfig:
    pass

class ModelTraining:
    def __init__(self):
        pass

    def initiate_modeltraining(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise CustomException(e,sys)