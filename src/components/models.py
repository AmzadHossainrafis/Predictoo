import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers





def LstmModel(trainX):
    model = keras.Sequential()
    model.add(layers.LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')

    return model


def Conv1DModel(trainX):
    model = keras.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model



def ConvLstmModel(trainX):
    model = keras.Sequential()
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(trainX.shape[1], 1, trainX.shape[2], trainX.shape[3])))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model                                

def bidirectionalLstmModel(trainX):
    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(50, activation='relu'), input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model

def Lstmseq2seqModel(trainX):
    model = keras.Sequential()
    model.add(layers.LSTM(50, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(layers.LSTM(50, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model    


def GruModel(trainX):
    model = keras.Sequential()
    model.add(layers.GRU(50, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def GruSeq2SeqModel(trainX): 
    model = keras.Sequential()
    model.add(layers.GRU(50, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(layers.GRU(50, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def Conv1d(trainX):
     model = keras.Sequential()
     model.add(layers.Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])) )
     model.add(layers.MaxPooling1D(pool_size=2))
     model.add(layers.Flatten())
     model.add(layers.Dense(50, activation='relu'))
     model.add(layers.Dense(1))
     model.compile(optimizer='adam', loss='mse')
     return model


def Convlstm(trainX):
        model = keras.Sequential()
        model.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.LSTM(50, activation='relu', return_sequences=True))
        model.add(layers.LSTM(50, activation='relu', return_sequences=False))
        model.add(layers.Flatten())
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse')

        return model

def transform_model(trainX):
    pass



model_list = { 'LstmModel': LstmModel,
            'Conv1DModel': Conv1DModel,
            'ConvLstmModel': ConvLstmModel,
            'bidirectionalLstmModel': bidirectionalLstmModel,
            'Lstmseq2seqModel': Lstmseq2seqModel,
            'GruModel': GruModel,
            'GruSeq2SeqModel': GruSeq2SeqModel,
            'Conv1d': Conv1d,
            'Convlstm': Convlstm

                }