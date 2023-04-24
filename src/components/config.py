from dataclasses import dataclass 
import os

@dataclass
class DataInjectionConfig:
    """Class to hold data injection configuration parameters"""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    row_data_path: str = os.path.join('artifacts', 'data.csv')

@dataclass
class ModelConfig: 
    """Class to hold model training configuration parameters"""
    model_name: str = 'GruModel'
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
class DataInjectionConfig:
    """Class to hold data injection configuration parameters"""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    row_data_path: str = os.path.join('artifacts', 'data.csv')


