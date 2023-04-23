import os
import sys
import pandas as pd
import numpy as np
from models import *
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
#callbacks 
import tensorflow as tf 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau 


from src.components.data_transformation import DataTransformation 
from src.components.data_transformation import DataTransformationConfig




@dataclass
class ModelConfig: 
    """Class to hold model training configuration parameters"""
    model_name: str = 'test_model'
    model_path: str = os.path.join('artifacts', 'model.pkl')
    model_actication: str = 'relu' 
    model_input_shape = None
    model_callback = None


@dataclass
class TraingConfig:
    """Class to hold model training configuration parameters"""
    epochs: int = 10 
    batch_size: int = 32 
    validation_split: float = 0.2 
    metrics = ['mae', 'mse', 'mape']
    optimizer = 'adam' 
    loss = 'binary_crossentropy' 
    learning_rate = 0.001 

@dataclass 
class Data_preprocessConfig:
    """Class to hold data preprocess configuration parameters"""   
    n_days_past =14
    n_days_future = 1
    n_features = 6   

@dataclass
class ModelTraining:
    def __init__(self) -> None:
        self.model_training_config = ModelConfig() 
        self.trainng_config = TraingConfig() 
        self.data_preprocessConfig = Data_preprocessConfig()


       
    def initiate_model_training(self,train_data,test_data) -> None:
        try:
            #data 
            trainX = pd.read_csv(train_data)
            #preprocess data 


            model_collection = model_list[self.model_training_config.model_name]
            model = model_collection(trainX)
            metrics = self.trainng_config.metrics
            model.compile(optimizer=self.trainng_config.optimizer, loss=self.trainng_config.loss, metrics=metrics)


            #train model 
            history=model.fit(train_data, epochs= self.trainng_config.epochs, validation_data=test_data, callbacks=self.model_training_config.model_callback) 
            
            
            #evaluate model 
            test_loss, test_acc = model.evaluate(test_data)

            logging.info(f"------------------------------------------------------------")
            logging.info(f"Model name: {self.model_training_config.model_name}")
            logging.info(f"Epochs: {self.trainng_config.epochs}")
            logging.info(f"Batch size: {self.trainng_config.batch_size}")
            logging.info(f"Validation split: {self.trainng_config.validation_split}")
            logging.info(f"Metrics: {self.trainng_config.metrics}")
            logging.info(f"Optimizer: {self.trainng_config.optimizer}")
            logging.info(f"Loss: {self.trainng_config.loss}")
            logging.info(f"Test loss: {test_loss}")
            logging.info(f"Test accuracy: {test_acc}")
            logging.info(f"Model training completed successfully")
            logging.info(f"------------------------------------------------------------")

            return history

        except Exception as e:
            logging.error(f"Exception occured while training model: {e}")
            raise CustomException(e,sys)
        



if __name__ == '__main__': 
    model_training = ModelTraining('test_model')
    model_training.initiate_model_training() 


