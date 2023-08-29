from dataclasses import dataclass 
import os
from src.components.matrices import r2_score 
import datetime 

@dataclass
class DataInjectionConfig:
    """Class to hold data injection configuration parameters"""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    row_data_path: str = os.path.join('artifacts', 'data.csv')
    dataset_path: str =  r'C:\Users\Amzad\Desktop\sqph_stock_prediction\notebook\data\Sqph_dataset.csv'

@dataclass
class ModelConfig: 
    """Class to hold model training configuration parameters"""
    model_name: str =  'LstmModel'
    model_path: str = os.path.join(r'C:\Users\Amzad\Desktop\PREDICTOO\artifacts\model_ckpt', '{}.h5'.format(model_name))
    model_actication: str = 'relu' 
    model_input_shape = None
    model_callback = None
    monitor = 'val_loss'
    verbose = 1
    save_best_only = True


#BiGru ,LstmModel ,BiGruSeq2Seq

@dataclass
class TraingConfig:
    """Class to hold model training configuration parameters"""
    epochs: int = 20
    batch_size: int = 128
    validation_split: float = 0.2 
    metrics = ['mae', 'mse', 'mape', r2_score]
    optimizer = 'adam' 
    loss = 'mse'
    learning_rate = 0.001 

@dataclass 
class Data_preprocessConfig:
    """Class to hold data preprocess configuration parameters"""   
    n_days_past =30
    n_days_future = 3
    #n_features = 6   



@dataclass
class DataInjectionConfig:
    """Class to hold data injection configuration parameters"""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    row_data_path: str = os.path.join('artifacts', 'data.csv')


@dataclass 
class adjustConfig:
    """Class to hold data preprocess configuration parameters"""   
    model_dir = r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\model_ckpt' 
    #list of model name 
    model_name = ['BiGruSeq2Seq','BiGru','LstmModel', 'BiGruSeq2Seq']


@dataclass 
class transformConfig:
    """Class to hold data preprocess configuration parameters"""   
    num_heads :int = 20
    ff_dim :int  = 256 
    num_layers :int = 5

@dataclass
class feature_selectionConfig:
    """Class to hold data preprocess configuration parameters"""   

    list_of_basic_features  =['Date','Open', 'High', 'Low', 'Close', 'Volume',]
    list_of_fandamental_features  =['Category_A_Volume','Sector_Volume','Price/EPS','Current_Share_Price/NAV'] 
    list_of_market_features  =['DSES', 'DS30', 'DSEX','DSES Index Changed', 'DS30 Index Changed', 'DSEX Index Changed','Beta',]
    list_of_economical_features=['Foreign_exchange_Buy', 'Foreign_exchange_rate_Sell','Gold Rate BD', 'Reserve in USD','Gold_Close_Price']
    list_of_scrip_features  =['Year_High','Year_Low','5nn_close_price_avg', '1st Support', '2nd Support','Ranking_by_sector_volume']

    list_of_tachnical_indicator  = [ 'SMA', 'SMA_10', 'SMA_20', 'SMA_50',
       'EMA_10', 'RSI', 'EMA_20', 'EMA_50', 'Breakout', 'Consolidate',
       'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'STOCH_SLOWK',
       'STOCH_SLOWD', 'CDLSHORTLINE', 'CDLRICKSHAWMAN', 'CDLSPINNINGTOP',
       'CDLINVERTEDHAMMER', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
       'CDLMATCHINGLOW', 'CDLGRAVESTONEDOJI', 'CDLHARAMI', '  ',
       'CDLHIKKAKE', 'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLDOJI',
       'CDLENGULFING']

    num_of_features : int = 5
    num_of_best_features : int = 5
    combine_features = list_of_basic_features 

@dataclass 
class PcaConfig:
    """Class to hold data preprocess configuration parameters"""   

    n_components : int = 5
    whiten : bool = True
    svd_solver : str = 'auto'
    tol : float = 0.0
    iterated_power : str = 'auto'
    random_state : int = None
    


# 