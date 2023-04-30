import sys 
import numpy as np 
import pandas as pd
from dataclasses import dataclass 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from exception import CustomException 
from src.logger import logging 
import os 







class DataTransformationConfig:
    """Class to hold data preprocess configuration parameters"""
    preprocessor_obj_file_path: str  = os.path.join('artifacts', 'preprocessor_obj.pkl')
    n_days_past: int = 14
    n_days_future : int  = 1

    
class DataTransformation:
    def __init__(self) -> None:
        # self.data_injection = data_injection
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try : 
            num_comumns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Year_High',
            'Year_Low', 'Current_Share_Price/NAV', 'Pivot Point', 'R1', 'R2',
            '1st Support', '2nd Support', 'Cash_Dividend', 'Dividend Yield', 'EPS',
            'Price/EPS', 'DSES', 'DS30', 'DSEX', 'Category_A_Volume',
            'DSES Index Changed', 'DS30 Index Changed', 'DSEX Index Changed',
            'Ranking_by_sector_volume', 'Gold_Close_Price', 'Sector_Volume',
            'Foreign_exchange_Buy', 'Foreign_exchange_rate_Sell',
            '5nn_close_price_avg', 'Year High', 'Corr1', 'Corr2', 'Beta',
            'Gold Rate BD', 'Reserve in USD', 'SMA', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_10', 'RSI', 'EMA_20', 'EMA_50', 'Breakout', 'Consolidate',
            'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'STOCH_SLOWK',
            'STOCH_SLOWD', 'CDLSHORTLINE', 'CDLRICKSHAWMAN', 'CDLSPINNINGTOP',
            'CDLINVERTEDHAMMER', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLGRAVESTONEDOJI', 'CDLHARAMI', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLDOJI',
            'CDLENGULFING']


           # cat_columns = ['Sector', 'Industry', 'Company Name'] 

            # training pipeline 

            num_pipeline = Pipeline(
                steps=[
            
                    ('imputer', SimpleImputer(strategy='median')),
                    ('std_scaler', StandardScaler())

                ]
                )
            
            # there is no cat columns in this dataset 
            # cat_pipeline = Pipeline( 
            #     steps=[
            #         ('imputer', SimpleImputer(strategy='most_frequent')),
            #         ('one_hot_encoder', OneHotEncoder())
            #         ('std_scaler', StandardScaler())
            #     ]
            # )

            preprocessor = ColumnTransformer(
            ['num_pipeline', num_pipeline, num_comumns],
           # ['cat_pipeline', cat_pipeline, cat_columns]
            )

            logging.info('Data transformation completed successfully')


            return preprocessor 

        except Exception as e:
            logging.error('Data transformation failed')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path, test_path):

        try:
            train_df = pd.read_csv(train_path) 
            test_df = pd.read_csv(test_path)
            logging.info('train test data loaded successfully')
            
            logging.info('Data transformation started')
            preprocessor_obj = self.get_data_transformer_object()
            target_column = 'Close'
            input_features_train_df = train_df
            input_features_test_df= test_df


            #train_df.drop(target_column, axis=1) 
            # target_train_df = train_df[target_column]
            #test_df.drop(target_column, axis=1)
            # target_test_df = test_df[target_column]

            
            input_features_test_df = preprocessor_obj.transform(input_features_test_df)

            logging.info('applying data transformation on train and test data') 

            
            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor_obj.transform(input_features_test_df)

            trainX = []
            trainY = []

            n_future = self.data_transformation_config.n_days_future   # Number of days we want to look into the future based on the past days.
            n_past = self.data_transformation_config.n_days_past  # Number of past days we want to use to predict the future.


            for i in range(n_past, len(input_features_train_arr) - n_future +1):
                trainX.append(input_features_train_arr[i - n_past:i, 0:input_features_train_arr.shape[1]])
                trainY.append(input_features_train_arr[i + n_future - 1:i + n_future, 0])

            train_arr, test_arr = np.array(trainX), np.array(trainY)
            print(train_arr.shape)
            print(test_arr.shape) 


            logging.info('saved preporcessor object')


            # save_object(

            #     filepath=self.data_transformation_config.preprocessor_obj_file_path,
            #     obj=preprocessor_obj 
            # )

            return (

                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.error('Data transformation failed')
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    obj= DataTransformation() 
    obj.initiate_data_transformation(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\train.csv', 
                                     r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\test.csv') 
    obj.get_data_transformer_object()

    