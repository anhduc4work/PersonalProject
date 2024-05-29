# adfuller library 
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt 
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import numpy as np

# check_adfuller
def check_adfuller(ts):
    # Dickey-Fuller test
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
# check_mean_std
def check_mean_std(ts):
    #Rolling statistics
    rolmean = ts.rolling(6).mean()
    rolstd = ts.rolling(6).std()
    plt.figure(figsize=(10,5))   
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()

def eda(ts, title = ''):
    f,ax = plt.subplots(1,3,figsize = (20,4), width_ratios=[2,1,1])
    ax[0].plot(ts)
    plot_acf(ts,lags=20, ax=ax[1])
    plot_pacf(ts,lags=20, ax=ax[2])
    check_adfuller(ts)
    ax[0].set_title(title)
    plt.show()


def valid_arima(data, target, order, n_forecast):

    X = data.index  
    y = data[target]

    # Cross validation
    score_mae = []
    score_mse = []
    N_SPLITS = 5
    folds = TimeSeriesSplit(n_splits=N_SPLITS)
    for i, (train_index, val_index) in enumerate(folds.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        y_train.index = X_train.values
        y_val.index = X_val.values

        model = ARIMA(y_train, order=order)
        model_fit = model.fit()
        # predict
        y_pred = model_fit.predict(len(y_train),len(y_train)+len(y_val)-1)

        # Calculate metrics
        score_mae.append(mean_absolute_error(y_val, y_pred))
        score_mse.append(mean_squared_error(y_val, y_pred))


    # Testing
    train = data[target][:-n_forecast]
    test = data[[target]][-n_forecast:]
    # fit model
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    # predict
    forecast = model_fit.predict(len(train),len(train)+len(test)-1)


    # Visualization
    f, ax = plt.subplots(1,2,figsize=(15,3),width_ratios=[3,1.2])

    pred = pd.Series(forecast.values,index=test.index)
    ax[0].plot(data[target].iloc[-n_forecast-30:],label = "original")
    ax[0].plot(pred,label = "predicted")
    ax[0].set_title("Time Series Forecast")
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel(target)
    print(f' MAE: {mean_absolute_error(test, forecast)}')
    print(f' MSE: {mean_squared_error(test, forecast)}')
    print(f' AIC: {model_fit.aic}')
    print(f' BIC: {model_fit.bic}')


    sns.lineplot(x=[str(i) for i in np.arange(1,N_SPLITS+1)], y=score_mae,  color='gold', label='MAE', ax=ax[1])
    sns.lineplot(x=[str(i) for i in np.arange(1,N_SPLITS+1)], y=score_mse, color='indianred', label='MSE', ax=ax[1])
    ax[1].set_title('Loss', fontsize=14)
    ax[1].set_xlabel(xlabel='Fold', fontsize=14)
    plt.show()