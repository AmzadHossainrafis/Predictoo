import os
from config import feature_selectionConfig
import pandas as pd 
#model evaluation function
from sklearn.preprocessing import StandardScaler
import numpy as np
import keras.backend as K
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.models import load_model
from bs4 import BeautifulSoup
import requests 
import json



def scraper(url="https://stocknow.com.bd/api/v1/instruments"):
    response = requests.get(url)
   
    soup = BeautifulSoup(response.text, 'html.parser')
    data = json.loads(soup.text)    

    ds30 = data['DS30']["close"]
    dses = data['DSES']["close"]
    dsex = data['DSEX']["close"]


    return ds30, dsex, dses




feature_selection = feature_selectionConfig()

    #r2 score function 






def prediciton(Open,High,Low,Close,Volume,Beta=0.74,Nav=118,eps=20.51,model_name="adjmodel"): 

    __model_dir_path = os.path.join(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts/model_ckpt', '{}.h5'.format(model_name))
    read_data = pd.read_csv(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\data.csv')
    Ds30, Dsex, Dses = scraper()
    #take last 13 days data
    read_data = read_data.tail(13)
    data= read_data[feature_selection.combine_features[1:]]
  
    #add new data at the end of the data frame 

    #claculate index change of Dses,Ds30,Dsex columns 
    Dses_change = abs((Dses - data['DSES'].iloc[-1])) #/data['DSES'].iloc[-1]
    Ds30_change = abs((Ds30 - data['DS30'].iloc[-1])) #/data['DS30'].iloc[-1]
    Dsex_change = abs((Dsex - data['DSEX'].iloc[-1])) #/data['DSEX'].iloc[-1]
    price_per_nav= data['Close'][-1]/Nav
    current_= data['Close'][-1]/eps
   
 
    data.loc[len(data.index)] = [Open,High,Low,Close,Volume,Dses,Ds30,Dsex,Dses_change,Ds30_change,Dsex_change,Beta]

    data = data.astype(float)
    scaler = StandardScaler()
    batch_of_data = scaler.fit_transform(data)
    #array of data 
    batch_of_data = np.array(batch_of_data) # batch, 14,5 
    #expand the dimension 
    batch_of_data = np.expand_dims(batch_of_data, axis=0) # 1, batch, 14,5
    model = load_model(__model_dir_path,compile=False )#custom_objects={'r2_score': r2_score})
    prediction = model.predict(batch_of_data)
    prediciton= np.repeat(prediction, 12, axis=1)
    prediciton = scaler.inverse_transform(prediciton)[:,2]
    price_per_nav = price_per_nav * prediciton
    current_ = current_ * prediciton

    
    return prediciton 





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


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - SS_res/(SS_tot + K.epsilon())

    
