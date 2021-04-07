import pandas as pd
import numpy as np
import matplotlib as plt
import quandl
from datetime import datetime

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
# rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import statsmodels.api as sm

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


def fetchSpecificData(*, startDate, endDate, queryParams, **kwargs):
    if validDate(startDate):
        pass
    else:
        return "invalid start date"

    if validDate(endDate):
        pass
    else:
        return "invalid end date"

    allDataColumns = ["BCHAIN/TOTBC", "BCHAIN/MKTCP", "BCHAIN/TRFEE", "BCHAIN/TRFUS", "BCHAIN/NETDF", "BCHAIN/NTRAN",
                      "BCHAIN/NTRAT", "BCHAIN/NTREP", "BCHAIN/NADDU", "BCHAIN/NTRBL", "BCHAIN/TOUTV", "BCHAIN/ETRAV",
                      "BCHAIN/ETRVU", "BCHAIN/TRVOU", "BCHAIN/TVTVR", "BCHAIN/MKPRU", "BCHAIN/CPTRV", "BCHAIN/CPTRA",
                      "BCHAIN/HRATE", "BCHAIN/MIREV", "BCHAIN/ATRCT", "BCHAIN/BCDDC", "BCHAIN/BCDDE", "BCHAIN/BCDDW",
                      "BCHAIN/BCDDM", "BCHAIN/BCDDY", "BCHAIN/BLCHS", "BCHAIN/AVBLS", "BCHAIN/MWTRV", "BCHAIN/MWNUS",
                      "BCHAIN/MWNTD", "BCHAIN/MIOPM", "BCHAIN/DIFF"]

    result = all(elem in allDataColumns for elem in queryParams)

    if result:
        pass
    else:
        return "invalid Query Parameters"

    todaysDate = datetime.today().strftime('%Y-%m-%d')
    quandl.ApiConfig.api_key = 'idjLfySzftDM7Cn1oSGi'
    data = quandl.get(queryParams, start_date=startDate, end_date=endDate)
    data.rename(columns=lambda x: x[0:12], inplace=True)

    data["Date"] = data.index

    data = data[data["Date"].notnull()]

    data.fillna(method='ffill', inplace=True)
    data = data.dropna(thresh=len(data) - 3, axis=1)  # to drop all columns where most of the entries are NAN

    return data

allData=fetchAllData()
data=fetchSpecificData(startDate="2020-03-01", endDate="2020-03-30",queryParams=["BCHAIN/MKPRU"])


import itertools
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    # print(param)
    for param_seasonal in seasonal_pdq:
        # print(param_seasonal)
        try:
            mod = sm.tsa.statespace.SARIMAX(allData["BCHAIN/MKPRU"],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except Exception as err:
            print(err)

            continue


mod = sm.tsa.statespace.SARIMAX(allData["BCHAIN/MKPRU"],
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

results.save('arima_model.pkl')


print("=====Done=====")
# load model
# loaded = ARIMAResults.load('model.pkl')

