import warnings
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pickle

import quandl
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
from datetime import datetime

import chart_studio.plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as pyoff


def parser(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')

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
    data=data.dropna(thresh=len(data) - 3, axis=1)

    return data

todaysDate=datetime.today().strftime('%Y-%m-%d')
data=fetchSpecificData(startDate="2015-01-01", endDate=todaysDate,queryParams=["BCHAIN/MKPRU"])
data = data.dropna()
Open = data[["BCHAIN/MKPRU"]]


def forecaster(Data, days_to_predict):
    Open = Data
    scaler = MinMaxScaler()
    scaler.fit(Open)
    train = scaler.transform(Open)

    n_input = days_to_predict
    n_features = 1

    #generate time series sequences for the forecast
    generator = TimeseriesGenerator(train, train, length=n_input, batch_size=8)

    model = Sequential()
    model.add(GRU(75, input_shape=(n_input, n_features)))
    model.add(Dense(1))
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    history = model.fit_generator(generator, epochs=100)

    # evaluate the model
    score = model.evaluate(generator)


    # serialize to JSON
    json_file = model.to_json()
    with open("model.json", "w") as file:
        file.write(json_file)
    # serialize weights to HDF5
    model.save_weights("model.hdf5")
    # model = Sequential()
    # model.add(LSTM(20, activation='relu', input_shape=(n_input, n_features)))
    # model.add(Dense(1))
    # model.compile(optimizer="nadam", loss="mse",metrics=['accuracy'])
    # history = model.fit_generator(generator,epochs=10)



    print("done")
    return "done"


forecaster(Open, 30)
