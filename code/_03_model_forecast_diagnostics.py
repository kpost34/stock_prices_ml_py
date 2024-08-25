# This script plots the forecasted model, performs diagnostics, and interprets the results


# Load Libraries, Functions, and Data===============================================================
## Load libraries
import pandas as pd
import numpy as np
import os
import pickle
import yfinance as yf
import matplotlib.pyplot as plt


## Load functions and 02 script
root = '/Users/keithpost/Documents/Python/Python projects/stock_prices_ml_py/'
os.chdir(root + 'code')
from _00_helper_fns import calc_mae, calc_rmse


## Data import
### Models
os.chdir(root + 'data')
file_mods = open('fitted_models.pkl', 'rb')
dict_models = pickle.load(file_mods)


### Training data
file_train = open('data_initial_clean.pkl', 'rb')
df0 = pickle.load(file_train)


### Test data
df_aapl_test = yf.download('AAPL', 
                           start='2019-01-01', 
                           end='2020-01-01')[['Adj Close']].rename(columns={"Adj Close": "aapl_adj_close"})
df_msft_test = yf.download('MSFT', 
                           start='2019-01-01', 
                           end='2020-01-01')[['Adj Close']].rename(columns={"Adj Close": "msft_adj_close"})
df_goog_test = yf.download('GOOG', 
                           start='2019-01-01', 
                           end='2020-01-01')[['Adj Close']].rename(columns={"Adj Close": "goog_adj_close"})
df_amzn_test = yf.download('AMZN', 
                           start='2019-01-01', 
                           end='2020-01-01')[['Adj Close']].rename(columns={"Adj Close": "amzn_adj_close"})



# Model Forecasting=================================================================================
## Extract individual models--------------------
model_aapl_fit = dict_models['aapl']
model_msft_fit = dict_models['msft']
model_amzn_fit = dict_models['amzn']
model_goog_fit = dict_models['goog']



## aapl--------------------
### Get forecasted values
forecast_aapl = model_aapl_fit.get_forecast(steps=21)

df_forecast_aapl_sum = forecast_aapl.summary_frame()
df_forecast_aapl_sum.index = df_aapl_test.iloc[:21].index


### Undo differencing and log transform
#get last known log-transformed value
last_aapl_log_value = np.log(df0['aapl_adj_close'].iloc[-1])

#undo differencing by calculating actual forecasted log-transformed values
df_aapl_log_forecast = df_forecast_aapl_sum.cumsum() + last_aapl_log_value

#undo log transform
df_aapl_forecast_values = np.exp(df_aapl_log_forecast)
df_aapl_forecast_values.rename(columns={col: f'aapl_{col}' for col in df_aapl_forecast_values}, 
                               inplace=True)


### Actual test values
df_aapl_test


### Combine actual and forecasted values
df_aapl_future_values = pd.concat([df_aapl_forecast_values, df_aapl_test], axis=1).iloc[0:21,]
df_aapl_future_values


### Plot forecasted vs true values
#### aapl
xlabs_forecast = ['2019-01-02', '2019-01-08', '2019-01-14', '2019-01-20', '2019-01-26', '2019-01-31']

plt.plot(df_aapl_future_values.index, df_aapl_future_values['aapl_mean'], color='green')
plt.plot(df_aapl_future_values.index, df_aapl_future_values['aapl_mean_ci_upper'], 
         color='green', linestyle='dashed') 
plt.plot(df_aapl_future_values.index, df_aapl_future_values['aapl_mean_ci_lower'],
         color='green', linestyle='dashed')
plt.plot(df_aapl_future_values.index, df_aapl_future_values['aapl_adj_close'])
plt.ylim(0, 70)
plt.xticks(xlabs_forecast)
plt.xlabel('Date')
plt.ylabel('Adjusted closing price')
plt.show()
plt.close()



## msft--------------------
### Get forecasted values
df_forecast_msft = model_msft_fit.get_forecast(steps=21)

df_forecast_msft_sum = df_forecast_msft.summary_frame()
df_forecast_msft_sum.index = df_msft_test.iloc[:21].index


### Undo differencing and log transform
#get last known log-transformed value
last_msft_log_value = np.log(df0['msft_adj_close'].iloc[-1])

#undo differencing by calculating actual forecasted log-transformed values
df_msft_log_forecast = df_forecast_msft_sum.cumsum() + last_msft_log_value

#undo log transform
df_msft_forecast_values = np.exp(df_msft_log_forecast) 
df_msft_forecast_values.rename(columns={col: f'msft_{col}' for col in df_msft_forecast_values}, 
                               inplace=True)


### Actual test values
df_msft_test


### Combine actual and forecasted values
df_msft_future_values = pd.concat([df_msft_forecast_values, df_msft_test], axis=1).iloc[0:21,]
df_msft_future_values



## amzn--------------------
### Get forecasted values
df_forecast_amzn = model_amzn_fit.get_forecast(steps=21)

df_forecast_amzn_sum = df_forecast_amzn.summary_frame()
df_forecast_amzn_sum.index = df_amzn_test.iloc[:21].index


### Undo differencing and log transform
#get last known log-transformed value
last_amzn_log_value = np.log(df0['amzn_adj_close'].iloc[-1])

#undo differencing by calculating actual forecasted log-transformed values
df_amzn_log_forecast = df_forecast_amzn_sum.cumsum() + last_amzn_log_value

#undo log transform
df_amzn_forecast_values = np.exp(df_amzn_log_forecast) 
df_amzn_forecast_values.rename(columns={col: f'amzn_{col}' for col in df_amzn_forecast_values}, 
                               inplace=True)


### Actual test values
df_amzn_test


### Combine actual and forecasted values
df_amzn_future_values = pd.concat([df_amzn_forecast_values, df_amzn_test], axis=1).iloc[0:21,]
df_amzn_future_values



## goog--------------------
### Get forecasted values
df_forecast_goog = model_goog_fit.get_forecast(steps=21)

df_forecast_goog_sum = df_forecast_goog.summary_frame()
df_forecast_goog_sum.index = df_goog_test.iloc[:21].index


### Undo differencing and log transform
#get last known log-transformed value
last_goog_log_value = np.log(df0['goog_adj_close'].iloc[-1])

#undo differencing by calculating actual forecasted log-transformed values
df_goog_log_forecast = df_forecast_goog_sum.cumsum() + last_goog_log_value

#undo log transform
df_goog_forecast_values = np.exp(df_goog_log_forecast) 
df_goog_forecast_values.rename(columns={col: f'goog_{col}' for col in df_goog_forecast_values}, 
                               inplace=True)


### Actual test values
df_goog_test


### Combine actual and forecasted values
df_goog_future_values = pd.concat([df_goog_forecast_values, df_goog_test], axis=1).iloc[0:21,]
df_goog_future_values



## Plot predicted vs actual values for all four stocks for January 2019--------------------
### Combine DFs
df_all_future_values = pd.concat([df_aapl_future_values, df_msft_future_values, df_amzn_future_values, 
                                  df_goog_future_values], axis=1)


### Plot
fig, axes = plt.subplots(2, 2)

t_mean_pred = df_all_future_values.filter(regex='_mean$').columns.tolist()
t_ci_u = df_all_future_values.filter(regex='_mean_ci_upper$').columns.tolist()
t_ci_l = df_all_future_values.filter(regex='_mean_ci_lower$').columns.tolist()
t_adj_close = df_all_future_values.filter(regex='adj_close$').columns.tolist()
t_stock = ['aapl', 'msft', 'amzn', 'goog']

xticks_forecast_all = ['2019-01-02', '2019-01-12', '2019-01-21', '2019-01-31']
xticklabs_forecast_all = ['01-02', '01-12', '01-21', '01-31'] 

n = 0

for i in range(0, 2):
  for j in range(0, 2):
    #extract strings from each list-string
    mean_pred = t_mean_pred[n]
    mean_ci_u = t_ci_u[n]
    mean_ci_l = t_ci_l[n]
    adj_close = t_adj_close[n]
    stock = t_stock[n]
    
    #generate lines
    axes[i, j].plot(df_all_future_values.index, df_all_future_values[adj_close], color='purple',
                    label='Actual value')
    axes[i, j].plot(df_all_future_values.index, df_all_future_values[mean_pred], color='blue', 
                    label='Predicted value')
    axes[i, j].plot(df_all_future_values.index, df_all_future_values[mean_ci_u], 
                   color='green', linestyle='dashed', label='Predicted CI') 
    axes[i, j].plot(df_all_future_values.index, df_all_future_values[mean_ci_l],
                   color='green', linestyle='dashed')
                   
    #set global limits
    axes[i, j].set_ylim(0, 200)
    
    #set labels for each subplot
    axes[i, j].set_xticks(xticks_forecast_all)
    axes[i, j].set_xticklabels(xticklabs_forecast_all)
    axes[i, j].set_title(stock)
    
    #increment n
    n = n + 1

#add super-labels
fig.supxlabel('Date')
fig.supylabel('Adjusted closing price')
fig.suptitle('Forecasted versus actual adjusted closing stock prices for four tech stocks \nin January 2019')

#add legend of just one set 
axes.flatten()[-2].legend(loc='lower center', bbox_to_anchor=(1, -0.2), ncol=3)

#adjust margins
plt.subplots_adjust(top=0.92, bottom=0.11)

plt.show()
plt.close()



# Model Diagnostics=================================================================================
## Mean absolute error (MAE)
df_mae_results = pd.concat([calc_mae(df_all_future_values, stock) for stock in t_stock])
df_mae_results
#MAE: provides a linear score that indicates how much, on average, the model's predictions deviate
  #from the actual values

#aapl, msft, and goog perform well: mae from ~$1-2 and mae_as_pct from ~2-3%
#amzn performs poorly: mae = $6 and mae_as_pct = 7.32%


## Root mean squared error (RMSE)
df_rmse_results = pd.concat([calc_rmse(df_all_future_values, stock) for stock in t_stock])
df_rmse_results
#RMSE: penalizes large errors more heavily than MAE; it's particularly useful when large errors
  #are undesirable (thus more sensitive to outliers)

#like mae, aapl, msft, and goog perform well: rmse from ~$1.5-2.5 and rmse_as_pct from ~2.5-4%
#amzn performs poorly: rmse = $6.41 and rmse_as_pct = 7.81%



# Interpretation====================================================================================
#see report










