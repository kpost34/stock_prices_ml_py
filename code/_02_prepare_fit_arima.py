# This script scales the target variable and fits a simple forecasting model


# Load Libraries and Data Import====================================================================
## Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import os


## Data import
#change wd
os.chdir(str(Path.cwd()) + '/data') #change wd
Path.cwd()

#import data
file = open('data_initial_clean.pkl', 'rb')
df0 = pickle.load(file)


# Data Wrangling====================================================================================
t_adj_close = ['aapl_adj_close', 'msft_adj_close', 'amzn_adj_close', 'goog_adj_close']
df_ac = df0[t_adj_close]



# Stationarity Check and Differencing===============================================================
## Stationarity
df_ac.apply(adfuller).iloc[1]
#no p-values < 0.5, so no series is stationary


## Differencing
df_ac_diff = df_ac.diff(axis=0).dropna()
df_ac_diff.apply(adfuller).iloc[1]
#all 0s, so all series are stationary


# Identify ARIMA Parameters (p, d,  q)==============================================================
## Autocorrelation (ACF)
#aapl
plt.figure(figsize=(12, 6))
plot_acf(df_ac_diff['aapl_adj_close'], lags=50, ax=plt.gca())
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Plot')
plt.show()
plt.close()
#y-axis = degree of autocorrelation
#x-axis = degree of lag; lag = 0 is unimportant because value is perfectly correlated with itself
#this plot shows the degree of correlation between time t and time t-0 -- t-50
#blue region = CI (which defaults at 95%)

#all four stocks
fig, axes = plt.subplots(2, 2)

t_stock = ['aapl', 'msft', 'amzn', 'goog']
n = 0

for i in range(0, 2):
  for j in range(0, 2):
    col = t_adj_close[n]
    stock = t_stock[n]
    
    plot_acf(df_ac_diff[col], ax=axes[i, j])
    axes[i, j].set_title(stock)
    
    n = n + 1

fig.subplots_adjust(top=1)
fig.suptitle("Autocorrelation of adjusted closing prices of \ntech stocks from 2020-2022")
fig.supxlabel('Lag')
fig.supylabel('Autocorrelation')
    
plt.tight_layout()
plt.show()
plt.close()
#strongest to weakest autocorrelation: goog, msft, aapl, amzn


## Partial Autocorrelation Function (PACF)



# Feature Scaling===================================================================================
t_adj_close = ['aapl_adj_close', 'msft_adj_close', 'amzn_adj_close', 'goog_adj_close']

### Normalized stock prices over time
#normalize data
df_ac = df[t_adj_close]

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_ac))
df_scaled.columns = t_adj_close
df_scaled.index = df.index
df_scaled



