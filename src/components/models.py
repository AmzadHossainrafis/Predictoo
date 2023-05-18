 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from config import transformConfig 




def LstmModel(trainX):
    model = keras.Sequential()
    model.add(layers.LSTM(32,activation= tf.keras.layers.LeakyReLU(alpha=0.2),input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Dense(1))
    #model.compile(loss='mae', optimizer='adam')

    return model


def Conv1DModel(trainX):
    model = keras.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')

    return model



def ConvLstmModel(trainX):
    model = keras.Sequential()
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(trainX.shape[1], 1, trainX.shape[2], trainX.shape[3])))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')

    return model                                

def bidirectionalLstmModel(trainX):
    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(32, activation='relu'), input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')

    return model

def Lstmseq2seqModel(trainX):
    model = keras.Sequential()
    model.add(layers.LSTM(32, activation=tf.keras.layers.LeakyReLU(alpha=0.2), input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(layers.LSTM(50, activation='relu'))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')

    return model    


def GruModel(trainX):
    model = keras.Sequential()
    model.add(layers.GRU(250, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')
    return model

def GruSeq2SeqModel(trainX): 
    model = keras.Sequential()
    model.add(layers.GRU(50, activation=tf.keras.layers.LeakyReLU(alpha=0.2), input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(layers.GRU(50, activation='relu'))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')
    return model

def Conv1d(trainX):
     model = keras.Sequential()
     model.add(layers.Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])) )
     model.add(layers.MaxPooling1D(pool_size=2))
     model.add(layers.Flatten())
     model.add(layers.Dense(50, activation='relu'))
     model.add(layers.Dense(1))
     #model.compile(optimizer='adam', loss='mse')
     return model


def Convlstm(trainX):
        model = keras.Sequential()
        model.add(layers.Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.LSTM(128, activation='relu', return_sequences=True))
        model.add(layers.LSTM(50, activation='relu', return_sequences=False))
        model.add(layers.Flatten())
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        #model.compile(optimizer='adam', loss='mse')

        return model

def BiGru(trainX):
    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.GRU(250, activation=tf.keras.layers.LeakyReLU(alpha=0.3)), input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')
    return model


def BiLstm(trainX):
    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(50, activation='relu'), input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')
    return model

def BiGruSeq2Seq(trainX):
    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.GRU(32, activation=tf.keras.layers.LeakyReLU(alpha=0.2), return_sequences=True), input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Bidirectional(layers.GRU(50, activation='relu')))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')
    return model

4

def transformer_encoder(trainX):
    transformer_encoder_config = transformConfig()
    num_heads=transformer_encoder_config.num_heads
    ff_dim=transformer_encoder_config.ff_dim
    num_layers=transformer_encoder_config.num_layers

    inputs = layers.Input((trainX.shape[1], trainX.shape[2]))
    x = inputs
    # positional embedding
    positions = tf.range(start=0, limit=trainX.shape[1], delta=1)
    positions = layers.Embedding(input_dim=trainX.shape[1], output_dim=trainX.shape[2])(positions)
    print(positions.shape)
    x += positions
 
    
    # Transformer Encoder
    for i in range(num_layers):
        # Multi-Head Attention
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=trainX.shape[2])(x, x)
        attn_output = layers.Dropout(0.1)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed Forward Network
        ffn = layers.Dense(ff_dim, activation='relu')(x)
        ffn = layers.Dense(trainX.shape[2])(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    # Output layer
    output = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(1)(output)

    # Define model
    model = keras.Model(inputs=inputs, outputs=output, name='transformer_encoder')
    return model

def Rnn(trainX):
    pass


model_list = { 'LstmModel': LstmModel,
            'Conv1DModel': Conv1DModel,
            'ConvLstmModel': ConvLstmModel,
            'bidirectionalLstmModel': bidirectionalLstmModel,
            'Lstmseq2seqModel': Lstmseq2seqModel,
            'GruModel': GruModel,
            'GruSeq2SeqModel': GruSeq2SeqModel,
            'Convlstm': Convlstm,
            'BiGru': BiGru ,
            'transform_model': transformer_encoder,
            'BiGruSeq2Seq': BiGruSeq2Seq

                }


