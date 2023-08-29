import numpy as np
import sys
import pandas as pd
import tensorflow as tf
from src.components.models import model_list
from sklearn.preprocessing import StandardScaler
from src.components.config import ModelConfig, TraingConfig, Data_preprocessConfig, feature_selectionConfig
from tensorflow.keras.callbacks import ModelCheckpoint
from src.components.model_evaluation import ModelEvaluations
from src.exception import CustomException
from src.logger import logging
from src.components.utils import prediciton , prediction_graph
import warnings
#from data_injection import DataInjection

warnings.filterwarnings('ignore')

import argparse 
parser = argparse.ArgumentParser() 
parser.add_argument("--n_past", help="number of past days ", type=int,default=14)
parser.add_argument("--n_feture", help="number of future days ", type=int,default=1)
args = parser.parse_args()

class ModelTraining:
    def __init__(self, save_model_fig=False):
        self.model_training_config = ModelConfig()
        self.trainng_config = TraingConfig()
        self.feature_selectionConfig = feature_selectionConfig()
        self.data_preprocessConfig = Data_preprocessConfig()
    
        self.save_model_fig = save_model_fig

    def trainner(self, train_dir):
        try:
            # Load and preprocess data
            trainX = pd.read_csv(train_dir)[self.feature_selectionConfig.combine_features]
            df_for_training = trainX.iloc[:, 1:].astype(float)
            scaler = StandardScaler().fit(df_for_training)
            df_for_training_scaled = scaler.transform(df_for_training)

            # Prepare training data
            trainX, trainY = [], []
            n_future = args.n_feture
            n_past = args.n_past

            for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
                trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
                #take n_future days close price in every n_past days and append it to trainY

                trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 3])


               
              

            trainX, trainY = np.array(trainX), np.array(trainY)
            print(trainX.shape, trainY.shape)

            # Create and compile model
            model = model_list[self.model_training_config.model_name](trainX , trainY)
            model.compile(optimizer=self.trainng_config.optimizer, loss=self.trainng_config.loss,
                          metrics=self.trainng_config.metrics)

            if self.save_model_fig:
                tf.keras.utils.plot_model(model,
                                          to_file=r'C:\Users\Amzad\Desktop\sqph_stock_prediction\figs/{}.png'.format(
                                              self.model_training_config.model_name),
                                          show_dtype=True,
                                          show_shapes=True,
                                          show_layer_names=True)

            # Train model
            logging.info("-------------------------new traing -------------------------")
            logging.info("Model training config: {}".format(self.trainng_config))
            logging.info('feature selection config: {}'.format(self.feature_selectionConfig))
            logging.info("Model config: {}".format(self.model_training_config))
            logging.info("-----------------------------------------------")

            history = model.fit(trainX, trainY, epochs=self.trainng_config.epochs,
                                batch_size=self.trainng_config.batch_size,
                                validation_split=self.trainng_config.validation_split,
                                callbacks=[ModelCheckpoint(self.model_training_config.model_path,
                                                           monitor=self.model_training_config.monitor,
                                                           verbose=1, save_best_only=self.model_training_config.save_best_only)])

            logging.info("--------------------------------------------------")
            logging.info("Model training completed")
            logging.info("--------------------------------------------------")

            return  model , trainX 

        except Exception as e:
            logging.error(f"Exception occurred while training model: {e}")
            raise CustomException(e, sys)



if __name__ == "__main__":

    # data_injection = DataInjection() <--- uncomment if your are training for 1st time
    # data_injection.initiate_data_injection() <--- uncomment if your are training for 1st time
    model_training = ModelTraining()
    # model_evaluation = ModelEvaluations()
    # model_evaluation.initiate_model_evaluation(
    #     r'C:\Users\Amzad\Desktop\PREDICTOO\artifacts\test.csv')

    model , trainX = model_training.trainner(r'C:\Users\Amzad\Desktop\PREDICTOO\artifacts\train.csv')
    prediction_graph(50,10, model , trainX , data_from='2021-2-1')
