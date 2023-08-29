# PREDICTOO (Predict the future of your time series data)
( under development , not ready for production , looking for contributors  )

## Description 

Predictoo is a python package which allows to predict the future of a time series.Predictoo contain 10 deep learning model .It is a tool for producing high quality forecasts for time series data that has multiple seasonality with linear or non-linear growth. It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays. It works best with time series that have strong seasonal effects and several seasons of historical data.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The things you need before installing the software.

* python 3.6 or higher 
* tensorflow 2.0 or higher 



### Installation

A step by step guide that will tell you how to get the development environment up and running.

bash  
```
$conda create -n predictoo python=3.6
$pip install -r requirements.txt 

```
ollah your good to go 



## how to train 
config the cofig.py accordingly to your training enviroment . Predictoo requre dataset ['Date','Open', 'High', 'Low', 'Close', 'Volume',] following formate .

for every time serice data there is a simple preprocessing pipeline which will formate your date and split the data into test , train inside the aftifact folder 


### must change in config.py
dataset_path: must reset according to your dataset dir 
 
model_path: must reset with a dir where you want to store your train weights 


```
$ python data_injections.py
```

bash  
```
$ cd src/components
$ python model_train.py -p 14 -f 1 

```
model evaluation 
```
$ python model_evaluation.py 
```


