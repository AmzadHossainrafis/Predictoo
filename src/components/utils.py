#model evaluation function
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import numpy as np
from config import * 



def test_evaluation():
    pass


def test_prediction():
    pass
    


def data_preprocess(data_dir ,num_cols, num_future_days, num_past_days):
    trainX = []
    trainY = []


    trainX = pd.read_csv(data_dir)
    #preprocess data 
    train_dates = pd.to_datetime(trainX['Date']) 

    cols = list(trainX)[1:num_cols]
    df_for_training = trainX[cols].astype(float)
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)

    #data transformation

    n_future = num_future_days  # Number of days we want to look into the future based on the past days.
    n_past = num_past_days  # Number of past days we want to use to predict the future.


    for i in range(n_past, len(df_for_training_scaled) - n_future +1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])


    return np.array(trainX), np.array(trainY)
#prediction function 