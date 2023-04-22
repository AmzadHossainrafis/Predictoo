import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#seed 
import numpy as np
np.random.seed(42)


@dataclass
class DataInjectionConfig:
    """Class to hold data injection configuration parameters"""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    row_data_path: str = os.path.join('artifacts', 'data.csv')


class DataInjection:
    def __init__(self) -> None:
        self.injection_config = DataInjectionConfig()

    def initiate_data_injection(self) -> None:
        """Initiate data injection"""
        logging.info('Initiating data injection')
        try:
            df = pd.read_csv(
                r'C:\Users\Amzad\Desktop\sqph_stock_prediction\notebook\data\Sqph_dataset.csv')
            logging.info('Data loaded successfully')
            os.makedirs(os.path.dirname(
                self.injection_config.train_data_path), exist_ok=True)
            logging.info('Artifats created successfully')
            df.to_csv(self.injection_config.row_data_path, index=False, header=True)

            # df.to_csv(self.injection_config.train_data_path, index=False, header=True)
            logging.info('Train test split initiated')

            # as this is a time serice data we can't use random split 
            train_set= df[ : int(len(df)*0.8)]
            test_set = df[int(len(df)*0.8) : ]
        
            train_set.to_csv(
                self.injection_config.train_data_path, index=False, header=True)
            logging.info('Train data saved successfully')
            test_set.to_csv(self.injection_config.test_data_path,
                            index=False, header=True)
            logging.info('data injection completed successfully')

            return{

                self.injection_config.train_data_path,
                self.injection_config.test_data_path,
            }

        except Exception as e:
            logging.error('Data loading failed')
            raise CustomException(e, sys)


if __name__ == '__main__':
    data_injection = DataInjection()
    data_injection.initiate_data_injection()