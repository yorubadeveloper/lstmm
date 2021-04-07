from flask import Flask, render_template
#from lstm import fetchAllData
import quandl
import numpy as np
import pandas as pd
from datetime import datetime
# , url_for
import warnings
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMAResults
import tensorflow as tf
from keras.optimizers import *
from keras.models import Sequential, model_from_json
from keras.layers import *
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pickle

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

    # generate time series sequences for the forecast
    generator = TimeseriesGenerator(train, train, length=n_input, batch_size=8)

    # load json and create model
    file = open("model.json", 'r')
    model_json = file.read()
    file.close()
    model = model_from_json(model_json)
    # load weights
    model.load_weights("model.hdf5")

    # evaluate the model

    # plt.figure(figsize=(15,5))
    # plt.plot(history.history['loss'])
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.show()

    # predict the data
    pred_train = model.predict(generator)
    pred_train = scaler.inverse_transform(pred_train)
    pred_train = pred_train.reshape(-1)

    pred_list = []

    batch = train[-n_input:].reshape((1, n_input, n_features))

    for i in range(n_input):
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:, 1:, :], [[pred_list[i]]], axis=1)

    add_dates = [Open.index[-1] + DateOffset(days=x) for x in range(0, days_to_predict + 1)]
    future_dates = pd.DataFrame(index=add_dates[1:], columns=Open.columns)

    # calculate the forecast
    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                              index=future_dates[-n_input:].index, columns=['Forecast'])

    df_predictions = pd.DataFrame(pred_train,
                                  index=Open.iloc[30:].index, columns=['Predictions'])

    df_proj = pd.concat([Open.iloc[30:], df_predict], axis=1)
    df_proj = pd.concat([df_proj, df_predictions], axis=1)
    df_proj = df_proj['2020-1-01':]

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
            y=df_proj['Predictions'],
            name='Prediction'
        )
    ]
    plot_layout = go.Layout(
        title='Bitcoin stock price prediction',
        template='plotly_dark'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    fig.write_html("templates/graph.html", auto_open=False)
    # pyoff.plot(fig, "file.html")

    new_data = pd.DataFrame(df_proj["Forecast"])

    new_data['Date'] = new_data.index

    new_data.tail(10)
    forecast = [x for x in new_data["Forecast"]][-(days_to_predict):]
    date = [x for x in new_data["Date"].dt.strftime('%Y-%m-%d')][-(days_to_predict):]
    res = {}
    for key in date:
        for value in forecast:
            res[key] = float(np.round(value, 2))
            forecast.remove(value)
            break
    return res


def validDate(date_text):
    import datetime
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def fetchAllData():

    allDataColumns = ["BCHAIN/TOTBC", "BCHAIN/MKTCP", "BCHAIN/TRFEE", "BCHAIN/TRFUS", "BCHAIN/NETDF",
                      "BCHAIN/NTRAN", "BCHAIN/NTRAT", "BCHAIN/NTREP", "BCHAIN/NADDU", "BCHAIN/NTRBL",
                      "BCHAIN/TOUTV", "BCHAIN/ETRAV", "BCHAIN/ETRVU", "BCHAIN/TRVOU", "BCHAIN/TVTVR",
                      "BCHAIN/MKPRU", "BCHAIN/CPTRV", "BCHAIN/CPTRA", "BCHAIN/HRATE", "BCHAIN/MIREV",
                      "BCHAIN/ATRCT", "BCHAIN/BCDDC", "BCHAIN/BCDDE", "BCHAIN/BCDDW", "BCHAIN/BCDDM",
                      "BCHAIN/BCDDY", "BCHAIN/BLCHS", "BCHAIN/AVBLS", "BCHAIN/MWTRV", "BCHAIN/MWNUS",
                      "BCHAIN/MWNTD", "BCHAIN/MIOPM", "BCHAIN/DIFF"]

    todaysDate = datetime.today().strftime('%Y-%m-%d')
    quandl.ApiConfig.api_key = 'idjLfySzftDM7Cn1oSGi'
    data = quandl.get(allDataColumns, start_date='2019-01-01', end_date=todaysDate)
    data.rename(columns=lambda x: x[0:12], inplace=True)

    data["Date"] = data.index
    data = data[data["Date"].notnull()]

    data.fillna(method='ffill', inplace=True)
    data = data.dropna(thresh=len(data) - 3, axis=1)  # to drop all columns where most of the entries are NAN

    return data


def arima_model(days):
    allData = fetchAllData()
    results = ARIMAResults.load('arima_model.pkl')
    pred = results.get_prediction(start=pd.to_datetime('2020-01-01'), dynamic=False)
    y_forecasted = pred.predicted_mean
    y_truth = allData["BCHAIN/MKPRU"]['2020-01-01':]
    pred_uc = results.get_forecast(steps=days)

    plot_data = [
        go.Scatter(
            x=allData["BCHAIN/MKPRU"]['2020-1-01':].index,
            y=allData["BCHAIN/MKPRU"]['2020-1-01':],
            name='Actual'
        ),
        go.Scatter(
            x=pred_uc.predicted_mean.index,
            y=pred_uc.predicted_mean,
            name='Prediction'
        ),
        go.Scatter(
            x=allData["BCHAIN/MKPRU"]['2020-1-01':].index,
            y=y_forecasted,
            name='Forecast'
        )
    ]
    plot_layout = go.Layout(
        title='Bitcoin stock price prediction'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    fig.write_html("templates/graph2.html", auto_open=False)
    return pred_uc


app = Flask(__name__)


@app.route('/')
def index():
    res = forecaster(Open, 30)

    print(res)

    response = arima_model(30)

    forecast = [x for x in response.predicted_mean]
    date = [x for x in response.predicted_mean.index.strftime('%Y-%m-%d')]
    arima_data = {}
    for key in date:
        for value in forecast:
            arima_data[key] = float(np.round(value, 2))
            forecast.remove(value)
            break


    print(arima_data)

    return render_template('index.html', res=res, data=res, arima=response)


@app.route('/graph')
def graph():
    return render_template('graph.html')


@app.route('/graph2')
def graph2():
    return render_template('graph2.html')


@app.route('/wallet/')
def wallet():
    return render_template('wallet.html')


if __name__ == '__main__':
    app.run(debug=True, port=5050)
