from config import Data_preprocessConfig, ModelConfig,TraingConfig
from models import model_list
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf 

class ModelEvaluations:
    def __init__(self) -> None:
        self.model_training_config = ModelConfig() 
        self.trainng_config= TraingConfig()
        self.data_preprocessConfig = Data_preprocessConfig()


       
    def initiate_model_evaluation(self,test_dir) -> None:
        try:
            #data  
            testX = pd.read_csv(test_dir)
            #preprocess data 
            train_dates = pd.to_datetime(testX['Date']) 

            cols = list(testX)[1:self.data_preprocessConfig.n_features]
            df_for_training = testX[cols].astype(float)
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
   
            #load model 
            model= model_list[self.model_training_config.model_name]
            model= model(trainX)
            model= tf.keras.models.load_model(self.model_training_config.model_path)





            metrics = self.trainng_config.metrics
            model.compile(optimizer=self.trainng_config.optimizer, loss=self.trainng_config.loss, metrics=metrics)
    
            model.evaluate(trainX,trainY) 

        except Exception as e:
            # logging.error(f"Exception occured while training model: {e}")
            # raise CustomException(e,sys)
            print(e)


if __name__ == "__main__": 
    model_evaluation = ModelEvaluations()
    model_evaluation.initiate_model_evaluation(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\test.csv')