 
from tensorflow import keras
from tensorflow.keras import layers
from models import model_list
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from models import model_list 
from config import ModelConfig, TraingConfig, Data_preprocessConfig, feature_selectionConfig

feature_selectionConfig = feature_selectionConfig()



df = pd.read_csv(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\data.csv')

train_dates = pd.to_datetime(df['Date']) 
df=df[feature_selectionConfig.combine_features]

cols = list(df)[1:]
df_for_training = df[cols].astype(float)
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

trainX = []
trainY = []

n_future = 1   
n_past = 14  

for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)







model_1 = model_list['LstmModel']
model_1= model_1(trainX)
#load the weights 
model1=model_1.load_weights(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\model_ckpt\LstmModel.h5')

#load the weights 
#model1=model_1.load_weights(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\model_ckpt\LstmModel.h5')
model1 = Sequential()
#freeze the layers and add the model to the sequential model 
for layer in model1.layers: 
    layer.trainable = False 
    model1.add(layer)


model_2 = model_list['BiGru']
model2= model_2(trainX)  
#load the weights
model2=model2.load_weights(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\model_ckpt\BiGru.h5')
model2 = Sequential()
#freeze the layers and add the model to the sequential model
for layer in model2.layers:
    layer.trainable = False 
    model2.add(layer)

model_3 = model_list['LstmModel']
model3= model_3(trainX)
#load the weights
model3=model3.load_weights(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\model_ckpt\LstmModel.h5')
model3 = Sequential()
#freeze the layers and add the model to the sequential model
for layer in model3.layers:
    layer.trainable = False 
    model3.add(layer)


model_4 = model_list['GruModel']
model4= model_4(trainX)
#load the weights
model4=model4.load_weights(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\model_ckpt\GruModel.h5')
#model4=tf.keras.saving.load_model(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\model_ckpt\Lstmseq2seqModel.h5')
model4 = Sequential()
#freeze the layers and add the model to the sequential model
for layer in model4.layers:
    layer.trainable = False 
    model4.add(layer)





def base_model(trainX,model1,model2,model3,model4):
    model_input = keras.Input(shape=(trainX.shape[1], trainX.shape[2]))
    output_1 = model1(model_input)
    output_2 = model2(model_input)
    output_3 = model3(model_input)
    output_4 = model4(model_input)

    #concatenate the output of the models
    output_1 = layers.Flatten()(output_1)
    output_2 = layers.Flatten()(output_2)
    output_3 = layers.Flatten()(output_3)
    output_4 = layers.Flatten()(output_4)
    output = layers.concatenate([output_1, output_2, output_3, output_4])

    #add adjusment layers 
    output = layers.Dense(128, activation='relu')(output)
    output = layers.Dense(50, activation='relu')(output)       
    output = layers.Dense(10, activation='relu')(output) 

    output = layers.Dense(1)(output)



    model = keras.Model(inputs=model_input, outputs=output)

    return model 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
    
model = base_model(trainX,model1,model2,model3,model4) 
model.summary()
model.compile(optimizer='adam', loss='mse',metrics=['mse'])
callbacks = [early_stopping]
tf.keras.utils.plot_model( model,
                        to_file=r'C:\Users\Amzad\Desktop\sqph_stock_prediction\figs/adjmodel.png',
                        show_shapes=True, show_layer_names=True)

# fit network
history = model.fit(trainX, trainY, epochs=50, batch_size=72, validation_split=0.2, verbose=2,callbacks=callbacks)
#save the model 
model.save(r'C:\Users\Amzad\Desktop\sqph_stock_prediction\artifacts\model_ckpt\adjmodel.h5')



from config import adjustConfig 

class AdjustmentModel: 
    def __init__(self):
        self.adjustConfig = adjustConfig() 
        self.model_dir = self.adjustConfig.model_dir 



    def base_model(self,model_list):

        for model in model_list: 
            model = model.load_weights(self.model_dir,compile=False)
            model = Sequential()
            for layer in model.layers: 
                layer.trainable = False 
                model.add(layer) 
            concate_layer = layers.concatenate(model)   

               
            
        return concate_layer 
    


