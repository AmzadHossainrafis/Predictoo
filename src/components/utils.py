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



feature_selection = feature_selectionConfig()


def prediciton(Open,High,Low,Close,Volume,model_name="adjmodel"): 

    __model_dir_path = os.path.join(r'C:\Users\Amzad\Desktop\PREDICTOO\artifacts/model_ckpt', '{}.h5'.format(model_name))
    read_data = pd.read_csv(r'C:\Users\Amzad\Desktop\PREDICTOO\artifacts\data.csv')
    path = r'C:\Users\Amzad\Desktop\PREDICTOO\notebook\data'
    

    Ds30, Dsex, Dses = scraper()
    #take last 13 days data
    read_data = read_data.tail(13)
    data= read_data[feature_selection.combine_features[1:]]
    

    path = r"C:\Users\Amzad\Downloads\Stocknow\Stocknow"
    result_df = read_stock_data(path)
    processed = calculate_returns(result_df)

    processed = processed.drop('Date', axis=1)
    processed.rename(columns={'Date(R)':'Date'}, inplace=True)

    filtered_df_SQURPH = pd.read_csv(r"C:\Users\Amzad\Downloads\SQ.csv") 
    Index = processed

    Final = calculate_beta(filtered_df_SQURPH, Index)
    Beta= Final['Beta'].iloc[-1]

    Dses_change = abs((Dses - data['DSES'].iloc[-1])) 
    Ds30_change = abs((Ds30 - data['DS30'].iloc[-1])) 
    Dsex_change = abs((Dsex - data['DSEX'].iloc[-1])) 

    data.loc[len(data.index)] = [Open,High,Low,Close,Volume,Dses,Ds30,Dsex,Dses_change,Ds30_change,Dsex_change,Beta]
    
    data = data.astype(float)
    scaler = StandardScaler()
    batch_of_data = scaler.fit_transform(data)
    #array of data 
    batch_of_data = np.array(batch_of_data) # batch, 14,5 
    #expand the dimension 
    batch_of_data = np.expand_dims(batch_of_data, axis=0) # 1, batch, 14,5
    model = load_model(__model_dir_path,compile=False )
    prediction = model.predict(batch_of_data)
    prediciton= np.repeat(prediction, 12, axis=1)
    prediciton = scaler.inverse_transform(prediciton)[:,3]+0.5
    return prediciton





def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - SS_res/(SS_tot + K.epsilon())

    



def read_stock_data(path):
    data_frames = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path)
            data_frames.append(df)
    return pd.concat(data_frames, axis=0, ignore_index=True)

def calculate_returns(df):
    df['Year'] = df['Date'].astype(str).str[:4].astype(int)
    df['Month'] = df['Date'].astype(str).str[4:6].astype(int)
    df['Day'] = df['Date'].astype(str).str[6:8].astype(int)

    #df['Date'] = pd.to_datetime(df['Date'])
    df['Date(R)'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    filtered_df = df[df["Scrip"] == "DSEX"]
    filtered_df = filtered_df.sort_values(by="Date(R)", ascending=False)
    filtered_df['Daily Return(DSEX)'] = filtered_df['Close'].pct_change()
    filtered_df['Variance'] = filtered_df['Daily Return(DSEX)'].var()
    return filtered_df


def calculate_beta(filtered_df_SQURPH, Index):
    
    filtered_df_SQURPH['Date'] = pd.to_datetime(filtered_df_SQURPH['Date'])
    filtered_df_SQURPH_M = pd.merge(filtered_df_SQURPH, Index[['Date', 'Daily Return(DSEX)', 'Variance']], on='Date', how='inner')
    filtered_df_SQURPH_M['Daily Return_SQ'] = filtered_df_SQURPH_M['Close'].pct_change()
    filtered_df_SQURPH_M['Covariance'] = filtered_df_SQURPH_M['Daily Return_SQ'].cov(filtered_df_SQURPH_M['Daily Return(DSEX)'])
    filtered_df_SQURPH_M['Beta'] = filtered_df_SQURPH_M['Covariance'] / filtered_df_SQURPH_M['Variance']

    return filtered_df_SQURPH_M


def calculate_yearly_high_low(df, date_column, high_column, low_column, close_column, start_date):
    new_df = df.copy()
    new_df[date_column] = pd.to_datetime(new_df[date_column])
    new_df = new_df.loc[new_df[date_column] >= start_date]
    new_df['Year'] = new_df[date_column].dt.year
    max_high_per_year = new_df.groupby('Year')[high_column].max()
    min_low_per_year = new_df.groupby('Year')[low_column].min()
    new_df['Max High'] = new_df['Year'].map(max_high_per_year)
    new_df['Min Low'] = new_df['Year'].map(min_low_per_year)
    new_df['Year_High'] = new_df['Max High'] - new_df[close_column]
    new_df['Year_Low'] = new_df['Min Low'] - new_df[close_column]
    
    return new_df[['Year_High', 'Year_Low', *new_df.columns]]





def calculate_support_resistance(processed_df2):
    processed_df2['Pivot Point'] = (processed_df2['High'] + processed_df2['Low'] + processed_df2['Close']) / 3
    processed_df2['R1'] = (processed_df2['Pivot Point'] * 2) - processed_df2['Low']
    processed_df2['R2'] = processed_df2['Pivot Point'] + (processed_df2['High'] - processed_df2['Low'])
    processed_df2['1st Support'] = (processed_df2['Pivot Point'] * 2) - processed_df2['High']
    processed_df2['2nd Support'] = processed_df2['Pivot Point'] - (processed_df2['High'] - processed_df2['Low'])
    return processed_df2


def stocknow(url):
    r = requests.get(url)
    data = r.json()
    return data

def merge_stock_data(stock_price_df, nav_url):
    # Fetch NAV data
    nav_data = stocknow(nav_url)
    nav_df = pd.DataFrame(nav_data['net_asset_val_per_share'])
    nav_df = nav_df.loc[nav_df['meta_date'] >= '2011-06-30']
    nav_df.rename(columns={'meta_date': 'Date', 'meta_value': 'Net Asset Value'}, inplace=True)
    nav_df['Date'] = pd.to_datetime(nav_df['Date'])

    # Merge stock price and NAV data
    merged_df = pd.merge(stock_price_df, nav_df, on="Date", how="left")
    merged_df = merged_df.sort_values("Date")

    for index, row in nav_df.iterrows():
        start_date = row['Date']
        end_date = nav_df.iloc[index + 1]['Date'] if index + 1 < len(nav_df) else pd.Timestamp.today().strftime('%Y-%m-%d')
        nav_value = row['Net Asset Value']
        merged_df.loc[(merged_df['Date'] >= start_date) & (merged_df['Date'] < end_date), 'Net Asset Value'] = nav_value

    merged_df['Close'] = pd.to_numeric(merged_df['Close'], errors='coerce')
    merged_df['Net Asset Value'] = pd.to_numeric(merged_df['Net Asset Value'], errors='coerce')
    merged_df['Current_Share_Price/NAV'] = merged_df['Close'] / merged_df['Net Asset Value']

    return merged_df



def scraper(url="https://stocknow.com.bd/api/v1/instruments"):
    response = requests.get(url)
   
    soup = BeautifulSoup(response.text, 'html.parser')
    data = json.loads(soup.text)    
    ds30 = data['DS30']["close"]
    dses = data['DSES']["close"]
    dsex = data['DSEX']["close"]
   
    return ds30, dsex, dses




def prediction_graph(n_days ,n_past ,models ,trainX, unit_of_chg=0.5, data_from='2022-2-1',title='bdcom prediction'):
    
    import numpy as np
    import seaborn as sns 
    import matplotlib.pyplot as plt 
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    import datetime as dt

    df = pd.read_csv(r'C:\Users\Amzad\Desktop\PREDICTOO\artifacts\train.csv')
    df = df[df['Date'] != 0]
    #Separate dates for future plotting
    train_dates = pd.to_datetime(df['Date'])
    #Variables for training
    cols = list(df)[1:6]
    df_for_training = df[cols].astype(float)
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    n_past = n_past #past 60 days to predict the next day
    n_days_for_prediction=n_days #let us predict past 15 days

    predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq=us_bd).tolist()
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    model = models
    #Make priction
    prediction = model.predict(trainX[-n_days_for_prediction:]) 
    prediction = prediction * unit_of_chg 
    prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
    # Convert timestamp to date
    forecast_dates = []
    for time_i in predict_period_dates:
        forecast_dates.append(time_i.date())
        
    df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
    df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

    original = df[['Date', 'Close']]
    original['Date']=pd.to_datetime(original['Date'])
    original = original.loc[original['Date'] >= data_from]
    fig, ax = plt.subplots(figsize=(12, 6))  
    # plot style 
    ax.set_facecolor('xkcd:light grey')
    sns.set_style("darkgrid")
  
    plt.title(title,fontsize=18)
    sns.lineplot(x =original['Date'], y= original['Close'],linewidth=2.5, color='blue', label='Actual')
    sns.lineplot(x=df_forecast['Date'], y=df_forecast['Open'],linewidth=2.5, color='red', label='Predicted')

    #save the graph 

    plt.savefig(r'C:\Users\Amzad\Desktop\PREDICTOO\figs\{}.png'.format(str(dt.datetime.now().date())))

     