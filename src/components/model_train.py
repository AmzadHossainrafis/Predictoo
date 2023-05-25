import numpy as np
import sys
import pandas as pd
import tensorflow as tf
from models import model_list
from sklearn.preprocessing import StandardScaler
from config import ModelConfig, TraingConfig, Data_preprocessConfig, feature_selectionConfig
from tensorflow.keras.callbacks import ModelCheckpoint
from model_evaluation import ModelEvaluations
from exception import CustomException
from logger import logging
from utils import prediciton
import warnings

warnings.filterwarnings('ignore')


class ModelTraining:
    def __init__(self, save_model_fig=False):
        self.model_training_config = ModelConfig()
        self.trainng_config = TraingConfig()
        self.data_preprocessConfig = Data_preprocessConfig()
        self.feature_selectionConfig = feature_selectionConfig()
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
            n_future = self.data_preprocessConfig.n_days_future
            n_past = self.data_preprocessConfig.n_days_past

            for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
                trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
                trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

            trainX, trainY = np.array(trainX), np.array(trainY)

            # Create and compile model
            model = model_list[self.model_training_config.model_name](trainX)
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

            return history

        except Exception as e:
            logging.error(f"Exception occurred while training model: {e}")
            raise CustomException(e, sys)



if __name__ == "__main__":

    # data_injection = DataInjection() <--- uncomment if your are training for 1st time
    # data_injection.initiate_data_injection() <--- uncomment if your are training for 1st time
    # model_training = ModelTraining()
    # model_training.trainner(
    #     r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\train.csv')
    # model_evaluation = ModelEvaluations()
    # model_evaluation.initiate_model_evaluation(
    #     r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\test.csv')
    #model_name= ModelConfig().model_name
    # result = prediciton(model_name , 221.2,221.2,219.4,219.7,218896)
    # logging.info("--------------------------------------------------")
    # logging.info("Prediction result on random value : {}".format(result))
    # print(result)

    pass        