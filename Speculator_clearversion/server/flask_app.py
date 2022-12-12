
# A very simple Flask Hello World app for you to get started with...

from flask import Flask
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def get_data_test(s1, window):
  data = get_technical_indicators(s1[['<CLOSE>','<HIGH>','<LOW>']])
  data = data.dropna()
  l = ['<CLOSE>','<HIGH>','<LOW>','20ema','50ema','k','d']


  data_test = []
  for i in l:
    val = data[[i]].values
    data_test.append(val)

  scaled_price_test = []
  scaled_indicators_test = []
  for i in range(len(data_test)):
    if (i==0):
      scalerx = MinMaxScaler()
      scalerx = scalerx.fit(data_test[i])
      scaled_price_test.append(scalerx.transform(data_test[i]))
      scaled_indicators_test.append(scalerx.transform(data_test[i]))
    else:
      scaler = MinMaxScaler()
      scaler = scaler.fit(data_test[i])
      scaled_indicators_test.append(scaler.transform(data_test[i]))
      a2,b2,c2 = trans(scaled_price_test,scaled_indicators_test,data_test[0])
  return a2,b2,c2,scalerx

def trans(scaled_price,scaled_indicators,arr):
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

def imitation(model1,a,b,scal):
  ans = model1.predict([a,b])
  for i in range(len(ans)):
    ans[i] = round(ans[i][0])
  a=a.reshape(a.shape[0],a.shape[1])
  c = scal.inverse_transform(a)
  return ans[-1],c[-1]


import os
from flask_cors import CORS



app = Flask(__name__)
CORS(app)
window = 48
@app.route('/')
def hello_world():

    model1 = keras.models.load_model(os.getcwd()+"/mysite/models")


    indexs = ["IBM","EBAY","AAPL","INTC"]
    s = ""
    for i in indexs:
        #Получение данных с биржи
        s+=i+";"
        print(i)
        data = pd.read_csv(
            'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+i+'&interval=5min&apikey=0XQ17ZF2A1FM3M4B&datatype=csv')

        data = data[::-1]
        print(data)
        #Переименование столбцов
        data.rename(columns = {'open':'<OPEN>', 'close':'<CLOSE>',

                                  'high':'<HIGH>','low':'<LOW>'}, inplace = True)
        ls_test, den_test, ans_test, scaler_test = get_data_test(data,window)
        ls_test=ls_test.reshape(ls_test.shape[0],ls_test.shape[1],1)
        pred, p1 = imitation(model1,ls_test, den_test, scaler_test)
        s+=str(p1[-1])
        s+=";"
        s+=str(round((p1[-1]-p1[-2])/p1[-2]*100,2))
        s+=";"
        if (pred == 1):
            s+="Up;"
        else:
            s+="Down;"
    return s

