import os
import sys
import numpy as np
import pandas as pd
from models import model_list
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from config import ModelConfig, TraingConfig ,Data_preprocessConfig
from tensorflow.keras.callbacks import ModelCheckpoint
# from src.logger import logging 
# from src.exception import CustomException
import tensorflow as tf 
from model_evaluation import ModelEvaluations 



class ModelTraining:
    def __init__(self) -> None:
        self.model_training_config = ModelConfig() 
        self.trainng_config = TraingConfig() 
        self.data_preprocessConfig = Data_preprocessConfig()


       
    def initiate_model_training(self,train_dir) -> None:
        try:
            #data 
            trainX = pd.read_csv(train_dir)
            #preprocess data 
            train_dates = pd.to_datetime(trainX['Date']) 

            cols = list(trainX)[1:self.data_preprocessConfig.n_features]
            df_for_training = trainX[cols].astype(float)
            scaler = StandardScaler()
            scaler = scaler.fit(df_for_training)
            df_for_training_scaled = scaler.transform(df_for_training)

            #data transformation
            trainX = []
            trainY = []

            n_future = self.data_preprocessConfig.n_days_future   # Number of days we want to look into the future based on the past days.
            n_past = self.data_preprocessConfig.n_days_past  # Number of past days we want to use to predict the future.


            for i in range(n_past, len(df_for_training_scaled) - n_future +1):
                trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
                trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

            trainX, trainY = np.array(trainX), np.array(trainY)
            model_collection = model_list[self.model_training_config.model_name]
            model = model_collection(trainX)
            metrics = self.trainng_config.metrics
            model.compile(optimizer=self.trainng_config.optimizer, loss=self.trainng_config.loss, metrics=metrics)
            #save model graph 
            #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            tf.keras.utils.plot_model( model,
                                         to_file=r'C:\Users\Amzad\Desktop\sqph_stock_prediction\figs/{}.png'.format(self.model_training_config.model_name),
                                         show_shapes=True, show_layer_names=True)

            #train model 
            history=model.fit(trainX,trainY ,epochs= self.trainng_config.epochs,
                              batch_size=self.trainng_config.batch_size,
                              validation_split=self.trainng_config.validation_split, 
                              callbacks=[ModelCheckpoint(self.model_training_config.model_path,
                                                          monitor=self.model_training_config.monitor,
                                                            verbose=1, save_best_only=self.model_training_config.save_best_only,
                                                              )]
                              ) 
            
            
            #evaluate model 
            #test_loss, test_acc = model.evaluate(test_data)
            # logging.info(f"------------------------------------------------------------")
            # logging.info(f"Model name: {self.model_training_config.model_name}")
            # logging.info(f"Epochs: {self.trainng_config.epochs}")
            # logging.info(f"Batch size: {self.trainng_config.batch_size}")
            # logging.info(f"Validation split: {self.trainng_config.validation_split}")
            # logging.info(f"Metrics: {self.trainng_config.metrics}")
            # logging.info(f"Optimizer: {self.trainng_config.optimizer}")
            # logging.info(f"Loss: {self.trainng_config.loss}")
            # logging.info(f"Learning rate: {self.trainng_config.learning_rate}")
            # # logging.info(f"Test loss: {test_loss}")
            # # logging.info(f"Test accuracy: {test_acc}")
            # logging.info(f"Model training completed successfully")
            # logging.info(f"------------------------------------------------------------")

            return history

        except Exception as e:
            # logging.error(f"Exception occured while training model: {e}")
            # raise CustomException(e,sys)
            print(e)
        


if __name__ == "__main__":

    
    model_training = ModelTraining()
    model_training.initiate_model_training(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\train.csv')
    model_evaluation = ModelEvaluations()
    model_evaluation.initiate_model_evaluation(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\test.csv')