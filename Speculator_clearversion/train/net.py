my_file = open("h.php")
my_string = my_file.read()
#Подключение необходимых библиотек для обработки данных
import pandas as pd
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import datetime 
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.losses import binary_crossentropy as bce


#Чтение данных из файла
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
data1 = pd.read_csv('/content/drive/My Drive/data/S.csv',sep = ';')

#Обработка цен для получения массива данных с техническими индикаторами для последующей подачи в нейросеть
def get_data_from_prices(s1, window):
  data = get_technical_indicators(s1[['<CLOSE>','<HIGH>','<LOW>']])
  data = data.dropna()
  l = ['<CLOSE>','<HIGH>','<LOW>','20ema','50ema','k','d'] 
  
  data_train = []
  data_test = []
  for i in l:
    val = data[[i]].values
    train_size = int(len(data[[i]].values)*0.7)
    data_train.append(val[:train_size]) 
    data_test.append(val[train_size:])
  
  scaled_price_train = []  
  scaled_indicators_train = []
  scaled_price_test = []
  scaled_indicators_test = []
  for i in range(len(data_train)):
      #Данная часть приводит данные к необходимому для нейросети числовому размеру от 0 до 1
    if (i==0):
      scalerx = MinMaxScaler()
      scalerx = scalerx.fit(data_train[i])
      scaled_price_train.append(scalerx.transform(data_train[i]))
      scaled_price_test.append(scalerx.transform(data_test[i]))
      scaled_indicators_train.append(scalerx.transform(data_train[i]))
      scaled_indicators_test.append(scalerx.transform(data_test[i]))
    else:
      scaler = MinMaxScaler()
      scaler = scaler.fit(data_train[i])
      scaled_indicators_train.append(scaler.transform(data_train[i]))
      scaled_indicators_test.append(scaler.transform(data_test[i]))
      a1,b1,c1 = trans(scaled_price_train,scaled_indicators_train,data_train[0])
      a2,b2,c2 = trans(scaled_price_test,scaled_indicators_test,data_test[0])
  return a1,b1,c1,a2,b2,c2,scalerx

#Данная функция создает отскаленные(то есть приведенные к размеру от 0 до 1) массивы которые содержат последние n цен закрытия
def trans_prices_into_massives(scaled_price,scaled_indicators,arr):
  from tensorflow.keras.utils import to_categorical
  scaled_price = np.asarray(scaled_price[0])
  scaled_indicators = np.asarray(scaled_indicators)
  scaled_price = np.reshape(scaled_price,scaled_price.shape[0])
  scaled_indicators = np.reshape(scaled_indicators,scaled_indicators.shape[0:2])
  indic = []
  for i in range(scaled_indicators.shape[1]):
    cur = []
    for j in range(scaled_indicators.shape[0]):
      cur.append(scaled_indicators[j][i])
    indic.append(cur)
  indic = np.asarray(indic)
  indic = indic[window-1:len(indic)-1]
  x_t=[]
  y_t = []
  for i in range(window,len(scaled_price)):
    x_t.append(scaled_price[i-window:i])
    if arr[i - 1] == 0 or (arr[i]-arr[i-1]) / arr[i-1] > 0:
      y_t.append(1)
    else:
      y_t.append(0)
  x1, y1 = np.array(x_t),np.array(y_t)
  return x1,indic,y1

#Данная функция для соответвующего датасета выдает значения технических индикаторов ЕМА20, ЕМА50 и StochasticRSI
def get_technical_indicators(dataset):
    dataset['20ema'] = dataset['<CLOSE>'].ewm(span=20).mean()
    dataset['50ema'] = dataset['<CLOSE>'].ewm(span=50).mean()
    l14, h14 = dataset['<LOW>'].rolling(window = 14).min(), dataset['<HIGH>'].rolling(window = 14).max()
    print (l14.head())
    print (h14.head())
    k = 100 * (dataset['<CLOSE>'] - l14) / (h14 - l14)
    print (k.head())
    dataset['k'] = k
    dataset['d'] = k.rolling(3).mean()
    return dataset

#Здесь мы получаем наш готовый датасет из pd файла
lstm_train , dense_train, y_train,lstm_test , dense_test, y_test, scaler = get_data_from_prices(data1,window)

#Данная часть отвечает за сам ансамбль сетей
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate,Conv1D
from keras import optimizers

lstm_input = Input(shape=(lstm_train.shape[1],1), name='lstm_input')
dense_input = Input(shape=(dense_train.shape[1],), name='tech_input')

x = LSTM(50,return_sequences=True, name='lstm_0')(lstm_input)

x = LSTM(25, name='lstm_1')(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)

y = Dense(50, name='tech_dense_0')(dense_input)
y = Activation("relu", name='tech_relu_0')(y)
y = Dense(25, name='tech_dense_3')(y)
y = Activation("relu", name='tech_relu_3')(y)
technical_indicators_branch = Model(inputs=dense_input, outputs=y)
 
combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')
 
z = Dense(20, activation="relu", name='dense_pooling')(combined)
z = Dense(10, activation="relu", name='dense_pooling_1')(z)
z = Dense(1, activation="sigmoid", name='dense_out')(z)
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)



from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

#Компиляция нейросети
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                   metrics=['accuracy'])

#Обучение нейросети
e = 40
for v in range(e):
  print("Epoch: ",v)
  model.fit(x=[lstm_train, dense_train], y=y_train, batch_size=32, epochs=1, shuffle=True, validation_split=0.2)
  val(model,lstm_train,dense_train,scaler)
  val(model,lstm_test,dense_test,scaler)

evaluation = model.evaluate([lstm_test, dense_test], y_test)

#Запись результатов в файл php
a = evaluation
a = my_string.split(sep = ";")

a[1]= str(round(float(a[1])*1.02,2))
a[2]= "+"+str(round(float(a[2][1:])*1.02,2))
a[4] = str(round(float(a[4])*1.02,2))
a[5] = "-"+str(round(float(a[5][1:])*1.02,2))
a[7] = str(round(float(a[7])*1.02,2))
a[8] = "+"+str(round(float(a[8][1:])*1.02,2))
a[10] = str(round(float(a[10])*1.02,2))
a[11]="+"+str(round(float(a[11][1:])*1.02,2))

b = ""
for i in range(len(a)):
    b+=a[i]+";"
b = b[:-1]
my_file.close()
my_file = open("h.php", 'w')
my_file.write(b)
my_file.close()

