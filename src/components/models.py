 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from config import transformConfig 

# class PositionalEncoding(layers.Layer):
#     def __init__(self, sequence_length, output_dim):
#         super(PositionalEncoding, self).__init__()
#         self.pos_encoding = self.positional_encoding(sequence_length, output_dim)

#     def get_angles(self, pos, i, output_dim):
#         angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(output_dim, tf.float32))
#         return pos * angle_rates

#     def positional_encoding(self, sequence_length, output_dim):
#         angle_rads = self.get_angles(
#             tf.expand_dims(tf.range(sequence_length, dtype=tf.float32), 1),
#             tf.expand_dims(tf.range(output_dim, dtype=tf.float32), 0),
#             output_dim,
#         )

#         sines = tf.math.sin(angle_rads[:, 0::2])
#         cosines = tf.math.cos(angle_rads[:, 1::2])
#         pos_encoding = tf.concat([sines, cosines], axis=-1)
#         pos_encoding = tf.expand_dims(pos_encoding, 0)
#         return tf.cast(pos_encoding, tf.float32)

#     def call(self, x):
#         return x + self.pos_encoding[:, :tf.shape(x)[1], :]


def LstmModel(trainX):
    model = keras.Sequential()
    model.add(layers.LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
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
    model.add(layers.Bidirectional(layers.LSTM(50, activation='relu'), input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')

    return model

def Lstmseq2seqModel(trainX):
    model = keras.Sequential()
    model.add(layers.LSTM(50, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(layers.LSTM(50, activation='relu'))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')

    return model    


def GruModel(trainX):
    model = keras.Sequential()
    model.add(layers.GRU(50, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')
    return model

def GruSeq2SeqModel(trainX): 
    model = keras.Sequential()
    model.add(layers.GRU(50, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
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
        model.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.LSTM(50, activation='relu', return_sequences=True))
        model.add(layers.LSTM(50, activation='relu', return_sequences=False))
        model.add(layers.Flatten())
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        #model.compile(optimizer='adam', loss='mse')

        return model

def BiGru(trainX):
    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.GRU(50, activation='relu'), input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(layers.Dense(1))
    #model.compile(optimizer='adam', loss='mse')
    return model





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





model_list = { 'LstmModel': LstmModel,
            'Conv1DModel': Conv1DModel,
            'ConvLstmModel': ConvLstmModel,
            'bidirectionalLstmModel': bidirectionalLstmModel,
            'Lstmseq2seqModel': Lstmseq2seqModel,
            'GruModel': GruModel,
            'GruSeq2SeqModel': GruSeq2SeqModel,
            'Conv1d': Conv1d,
            'Convlstm': Convlstm,
            'BiGru': BiGru ,
            'transform_model': transformer_encoder,

                }