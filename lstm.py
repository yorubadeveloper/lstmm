import warnings
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import optimizers
from keras.layers import LSTM, Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

warnings.filterwarnings("ignore")
from datetime import datetime

import chart_studio.plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as pyoff
import quandl


def parser(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')

def fetchAllData():
    allDataColumns=["BCHAIN/TOTBC","BCHAIN/MKTCP","BCHAIN/TRFEE","BCHAIN/TRFUS","BCHAIN/NETDF","BCHAIN/NTRAN","BCHAIN/NTRAT","BCHAIN/NTREP","BCHAIN/NADDU","BCHAIN/NTRBL","BCHAIN/TOUTV","BCHAIN/ETRAV","BCHAIN/ETRVU","BCHAIN/TRVOU","BCHAIN/TVTVR","BCHAIN/MKPRU","BCHAIN/CPTRV","BCHAIN/CPTRA","BCHAIN/HRATE","BCHAIN/MIREV","BCHAIN/ATRCT","BCHAIN/BCDDC","BCHAIN/BCDDE","BCHAIN/BCDDW","BCHAIN/BCDDM","BCHAIN/BCDDY","BCHAIN/BLCHS","BCHAIN/AVBLS","BCHAIN/MWTRV","BCHAIN/MWNUS","BCHAIN/MWNTD","BCHAIN/MIOPM","BCHAIN/DIFF"]
    todaysDate=datetime.today().strftime('%Y-%m-%d')
    quandl.ApiConfig.api_key = 'idjLfySzftDM7Cn1oSGi'
    data=quandl.get(allDataColumns,start_date ='2019-01-01', end_date = todaysDate)
    data.rename(columns=lambda x: x[0:12], inplace=True)
    data["Date"]=data.index
    data=data[data["Date"].notnull()]
    #print(data.isna().sum())
 
    data.fillna(method='ffill', inplace=True)
    data=data.dropna(thresh=len(data) - 3, axis=1) #to drop all columns where most of the entries are NAN
    print("==================")
 
    return data

def fetchSpecificData(*, startDate,endDate,queryParams, **kwargs):
    if parser(startDate):
        pass
    else:
        return "invalid start date"
            
    if parser(endDate):
        pass
    else:
        return "invalid end date"
            
    allDataColumns=["BCHAIN/TOTBC","BCHAIN/MKTCP","BCHAIN/TRFEE","BCHAIN/TRFUS","BCHAIN/NETDF","BCHAIN/NTRAN","BCHAIN/NTRAT","BCHAIN/NTREP","BCHAIN/NADDU","BCHAIN/NTRBL","BCHAIN/TOUTV","BCHAIN/ETRAV","BCHAIN/ETRVU","BCHAIN/TRVOU","BCHAIN/TVTVR","BCHAIN/MKPRU","BCHAIN/CPTRV","BCHAIN/CPTRA","BCHAIN/HRATE","BCHAIN/MIREV","BCHAIN/ATRCT","BCHAIN/BCDDC","BCHAIN/BCDDE","BCHAIN/BCDDW","BCHAIN/BCDDM","BCHAIN/BCDDY","BCHAIN/BLCHS","BCHAIN/AVBLS","BCHAIN/MWTRV","BCHAIN/MWNUS","BCHAIN/MWNTD","BCHAIN/MIOPM","BCHAIN/DIFF"]
          
    result =  all(elem in allDataColumns   for elem in queryParams  )
 
    if result:
        pass   
    else :
        return "invalid Query Parameters"
 
    todaysDate=datetime.today().strftime('%Y-%m-%d')
    quandl.ApiConfig.api_key = 'idjLfySzftDM7Cn1oSGi'
    data=quandl.get(queryParams,start_date =startDate, end_date = endDate)
    data.rename(columns=lambda x: x[0:12], inplace=True)
          
    data["Date"]=data.index
 
    data=data[data["Date"].notnull()]
    data.fillna(method='ffill', inplace=True)
    data=data.dropna(thresh=len(data) - 3, axis=1) #to drop all columns where most of the entries are NAN
 
    return data

todaysDate=datetime.today().strftime('%Y-%m-%d')
data=fetchSpecificData(startDate="2015-01-01", endDate=todaysDate,queryParams=["BCHAIN/MKPRU"])
data = data.dropna()
Open = data[["BCHAIN/MKPRU"]]

scaler = MinMaxScaler()
scaler.fit(Open)
train = scaler.transform(Open)
train.shape

n_input = 10
n_features = 1
 
#generate time series sequences for the forecast 
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=8)
#generator
#1
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer="nadam", loss="mse",metrics=['accuracy'])
history = model.fit_generator(generator,epochs=10)

#evaluate the model
score = model.evaluate(generator)
#print(score)

#plot loss over number of epochs
plt.figure(figsize=(15,5))
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

#predict the data
pred_train=model.predict(generator)
pred_train=scaler.inverse_transform(pred_train)
pred_train=pred_train.reshape(-1)

pred_list = []
#j
batch = train[-n_input:].reshape((1, n_input, n_features))
 
for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

add_dates = [Open.index[-1] + DateOffset(days=x) for x in range(0,11) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=Open.columns)

#calculate the forecast
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Forecast'])
 
df_proj = pd.concat([Open,df_predict], axis=1)
 
df_proj.tail(11)

plot_data = [
    go.Scatter(
        x=df_proj.index,
        y=df_proj['BCHAIN/MKPRU'],
        name='Actual'
    ),
    go.Scatter(
        x=df_proj.index,
        y=df_proj['Forecast'],
        name='Forecast'
    ),
      go.Scatter(
        x=df_proj.index,
        y=pred_train,
        name='Prediction'
    )
]
plot_layout = go.Layout(
        title='Bitcoin stock price prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.plot(fig, "file.html")
