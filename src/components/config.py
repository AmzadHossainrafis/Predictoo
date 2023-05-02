from dataclasses import dataclass 
import os
from utils import r2_score 

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
    model_path: str = os.path.join(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts/model_ckpt', '{}.h5'.format(model_name))
    model_actication: str = 'relu' 
    model_input_shape = None
    model_callback = None
    monitor = 'val_loss'
    verbose = 1
    save_best_only = True


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
    n_days_past =14
    n_days_future = 1
    n_features = 6   



@dataclass
class DataInjectionConfig:
    """Class to hold data injection configuration parameters"""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    row_data_path: str = os.path.join('artifacts', 'data.csv')


@dataclass 
class adjustConfig:
    """Class to hold data preprocess configuration parameters"""   
    pass




@dataclass 
class transformConfig:
    """Class to hold data preprocess configuration parameters"""   
    num_heads = 5 
    ff_dim = 256 
    num_layers = 5


