# This script scales the target variable and fits a simple forecasting model


# Load Libraries and Data Import====================================================================
## Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import yfinance as yf


## Data import
#change wd
root = '/Users/keithpost/Documents/Python/Python projects/stock_prices_ml_py/'
os.chdir(root + 'data') #change wd

#import training data
file = open('data_initial_clean.pkl', 'rb')
df0 = pickle.load(file)



# Data Transformation===============================================================================
## Log-transform the training data
t_adj_close = ['aapl_adj_close', 'msft_adj_close', 'amzn_adj_close', 'goog_adj_close']
df_ac = df0[t_adj_close]
df_ac_log = df_ac.apply(np.log)



# Stationarity Check and Differencing===============================================================
## Stationarity
df_adfuller_results = df_ac_log.apply(adfuller).iloc[1].reset_index().rename({'index': 'variable', 
                                                                              1: 'p-value'},
                                                                              axis=1)
df_adfuller_results 
#no p-values < 0.5, so no series is stationary


## Differencing
df_ac_log_diff = df_ac_log.diff(axis=0).dropna()
df_diff_adfuller_results = df_ac_log_diff.apply(adfuller).iloc[1].reset_index().rename({'index': 'variable', 
                                                                                        1: 'p-value'},
                                                                                        axis=1)
df_diff_adfuller_results                                                                                     
#all 0s, so all series are stationary


# Identify ARIMA Parameters (p, d,  q)==============================================================
## Autocorrelation (ACF)
#aapl
plt.figure(figsize=(12, 6))
plot_acf(df_ac_log_diff['aapl_adj_close'], lags=50, ax=plt.gca())
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Plot')
plt.show()
plt.close()
#y-axis = degree of autocorrelation
#x-axis = degree of lag; lag = 0 is unimportant because value is perfectly correlated with itself
#this plot shows the degree of correlation between time t and time t-0 -- t-50
#blue region = CI (which defaults at 95%)
#suggests q = 0 (or possibly q = 7)

#all four stocks
fig, axes = plt.subplots(2, 2)

t_stock = ['aapl', 'msft', 'amzn', 'goog']
n = 0

for i in range(0, 2):
  for j in range(0, 2):
    col = t_adj_close[n]
    stock = t_stock[n]
    
    plot_acf(df_ac_log_diff[col], ax=axes[i, j])
    axes[i, j].set_title(stock)
    
    n = n + 1

fig.subplots_adjust(top=1)
fig.suptitle("Autocorrelation of adjusted closing prices of \ntech stocks from 2016-2018")
fig.supxlabel('Lag')
fig.supylabel('Autocorrelation')
    
plt.tight_layout()

plot_initial_acf = plt.gcf()

plt.show()
plt.close()
#use ACF plot to determine q (MA order)
#q is the number of lags where the autocorrelation cuts off (drops sharply and stays close to 0)
#approximate values of q based on plots:
  #aapl: 0
  #msft: 1
  #amzn: 0
  #goog: 0


## Partial Autocorrelation Function (PACF)
#partial autocorrelation at lag k is the autocorrelation between x(t) and x(t-k) that is not 
  #accounted for by lags 1 through k-1
#critical difference between autocorrelation and partial autocorrelation is the inclusion/exclusion
  #of indirect correlations in the calculation
fig, axes = plt.subplots(2, 2)

t_stock = ['aapl', 'msft', 'amzn', 'goog']
n = 0

for i in range(0, 2):
  for j in range(0, 2):
    col = t_adj_close[n]
    stock = t_stock[n]
    
    plot_pacf(df_ac_log_diff[col], ax=axes[i, j])
    axes[i, j].set_title(stock)
    
    n = n + 1

fig.subplots_adjust(top=1)
fig.suptitle("Partial autocorrelation of adjusted closing prices of \ntech stocks from 2016-2018")
fig.supxlabel('Lag')
fig.supylabel('Partial autocorrelation')
    
plt.tight_layout()

plot_initial_pacf = plt.gcf()

plt.show()
plt.close()
#p is the number of lags before spikes drop to zero/near-zero
#estimated values using plots:
  #aapl: 0
  #msft: 2
  #amzn: 0
  #goog: 0
  
  
  
# Fit ARIMA Model===================================================================================
d = 1 #order of differencing for all four stocks

## aapl-------------------------
### First run
#### Set parameters
p = 0
q = 0


#### Fit model
model_aapl = ARIMA(df_ac_log_diff['aapl_adj_close'], order=(p, d, q))
model_aapl_fit = model_aapl.fit()
print(model_aapl_fit.summary()) 
#high degree of autocorrelation per Ljung-Box test result


### Second run
#### Set parameters
p = 1
q = 1


#### Fit model
model_aapl = ARIMA(df_ac_log_diff['aapl_adj_close'], order=(p, d, q))
model_aapl_fit = model_aapl.fit()
print(model_aapl_fit.summary()) 
#non-significant autocorrelation...but AR term is 'marginally' significant and MA term is significant
#drop AR term (p)


### Third run
#### Set parameters
p = 0 
q = 1


#### Fit model
model_aapl = ARIMA(df_ac_log_diff['aapl_adj_close'], order=(p, d, q))
model_aapl_fit = model_aapl.fit()
print(model_aapl_fit.summary()) 
#non-significant autocorrelation
#MA term is still significant (more parsimonious model)
#log likelihood and AIC relatively unchanged


#### Plot residuals
#### Assess normality and heteroscedasticity
resid_aapl = pd.DataFrame(model_aapl_fit.resid)

fig, ax = plt.subplots(1, 2)

ax[0].scatter(model_aapl_fit.fittedvalues, resid_aapl)
ax[0].axhline(0, color='red', linestyle='--')
ax[0].set_xlabel('Fitted Values')
ax[0].set_ylabel('Residuals')
ax[0].set_title('Residuals versus fitted values \n(Heteroscedasticity)')

resid_aapl.plot(ax=ax[1], kind='kde')
ax[1].set_xlabel('Residuals')
ax[1].set_ylabel('Density')
ax[1].set_title("Density of residuals \n(Normality)")
ax[1].get_legend().remove()

plt.tight_layout()

plot_aapl_var_norm = plt.gcf()

plt.show()
plt.close()
#homoscedastic: despite fitted values outliers, residuals are consistent throughout range
#normally distributed--symmetrical and somewhat bell-shaped curve

#understand residuals more
model_aapl_fit.fittedvalues.sort_values(ascending=True).head(30)
#very small residuals (high negative values) are at start of data and may indicate some type of
  #different phase or a model adjustment


#### Assess autocorrelation and partial autocorrelation
fig, ax = plt.subplots(1, 2)

plot_pacf(resid_aapl, ax=ax[0], title="")
ax[0].set_ylabel("Partial autocorrelation")

plot_acf(resid_aapl, ax=ax[1], title="")
ax[1].set_ylabel("Autocorrelation")

fig.supxlabel("Lag")

plt.tight_layout()

plot_aapl_resid_pacf_acf = plt.gcf()

plt.show()
plt.close()
#no concerns about autocorrelation of residuals



## msft-------------------------
### Set parameters
p = 2
q = 1


### Fit model
model_msft = ARIMA(df_ac_log_diff['msft_adj_close'], order=(p, d, q))
model_msft_fit = model_msft.fit()
print(model_msft_fit.summary())
#non-significant autocorrelation
#AR1, AR2, and MA terms significant


## Plot residuals
#### Assess normality and heteroscedasticity
resid_msft = pd.DataFrame(model_msft_fit.resid)

fig, ax = plt.subplots(1, 2)

ax[0].scatter(model_msft_fit.fittedvalues, resid_msft)
ax[0].axhline(0, color='red', linestyle='--')
ax[0].set_xlabel('Fitted Values')
ax[0].set_ylabel('Residuals')
ax[0].set_title('Residuals versus fitted values \n(Heteroscedasticity)')

resid_msft.plot(ax=ax[1], kind='kde')
ax[1].set_xlabel('Residuals')
ax[1].set_ylabel('Density')
ax[1].set_title("Density of residuals \n(Normality)")
ax[1].get_legend().remove()

plt.tight_layout()

plot_msft_var_norm = plt.gcf()

plt.show()
plt.close()
#strong evidence of homoskedasticity and normality of residuals


#### Assess autocorrelation and partial autocorrelation
fig, ax = plt.subplots(1, 2)

plot_pacf(resid_msft, ax=ax[0], title="")
ax[0].set_ylabel("Partial autocorrelation")

plot_acf(resid_msft, ax=ax[1], title="")
ax[1].set_ylabel("Autocorrelation")

fig.supxlabel("Lag")

plt.tight_layout()

plot_msft_resid_pacf_acf = plt.gcf()

plt.show()
plt.close()
#no concerns



## amzn-------------------------
### First run
#### Set parameters
p = 0
q = 0


### Fit model
model_amzn = ARIMA(df_ac_log_diff['amzn_adj_close'], order=(p, d, q))
model_amzn_fit = model_amzn.fit()
print(model_amzn_fit.summary())
#high autocorrelation


### Second run
#### Set parameters
p = 1
q = 1


### Fit model
model_amzn = ARIMA(df_ac_log_diff['amzn_adj_close'], order=(p, d, q))
model_amzn_fit = model_amzn.fit()
print(model_amzn_fit.summary())
#non-significant autocorrelation
#AR term non-significant
#MA term significant


### Third run
#### Set parameters
p = 0
q = 1


### Fit model
model_amzn = ARIMA(df_ac_log_diff['amzn_adj_close'], order=(p, d, q))
model_amzn_fit = model_amzn.fit()
print(model_amzn_fit.summary())
#non-significant autocorrelation
#MA term is non-signicant


### Fourth run (same as 2nd)
#### Set parameters
p = 1
q = 1


### Fit model
model_amzn = ARIMA(df_ac_log_diff['amzn_adj_close'], order=(p, d, q))
model_amzn_fit = model_amzn.fit()
print(model_amzn_fit.summary())
#non-significant autocorrelation
#AR term non-significant
#MA term significant
#howver, as shown in the third model, dropping the AR term leads to a model with a non-significant
  #parameter, so this model is the best choice


## Plot residuals
#### Assess normality and heteroscedasticity
resid_amzn = pd.DataFrame(model_amzn_fit.resid)

fig, ax = plt.subplots(1, 2)

ax[0].scatter(model_amzn_fit.fittedvalues, resid_amzn)
ax[0].axhline(0, color='red', linestyle='--')
# ax[0].set_xlim(0, 0.0022) #showws symmetrical distribution with low outliers
ax[0].set_xlabel('Fitted Values')
ax[0].set_ylabel('Residuals')
ax[0].set_title('Residuals versus fitted values \n(Heteroscedasticity)')

resid_msft.plot(ax=ax[1], kind='kde')
ax[1].set_xlabel('Residuals')
ax[1].set_ylabel('Density')
ax[1].set_title("Density of residuals \n(Normality)")
ax[1].get_legend().remove()

plt.tight_layout()

plot_amzn_var_norm = plt.gcf()

plt.show()
plt.close()
#strong evidence of homoskedasticity and normality of residuals


#### Assess autocorrelation and partial autocorrelation
fig, ax = plt.subplots(1, 2)

plot_pacf(resid_amzn, ax=ax[0], title="")
ax[0].set_ylabel("Partial autocorrelation")

plot_acf(resid_amzn, ax=ax[1], title="")
ax[1].set_ylabel("Autocorrelation")

fig.supxlabel("Lag")

plt.tight_layout()

plot_amzn_resid_pacf_acf = plt.gcf()

plt.show()
plt.close()
#no concerns



## goog-------------------------
### First run
#### Set parameters
p = 0
q = 0

#### Fit model
model_goog = ARIMA(df_ac_log_diff['goog_adj_close'], order=(p, d, q))
model_goog_fit = model_goog.fit()
print(model_goog_fit.summary())
#high autocorrelation


### Second run
#### Set parameters
p = 1
q = 1

#### Fit model
model_goog = ARIMA(df_ac_log_diff['goog_adj_close'], order=(p, d, q))
model_goog_fit = model_goog.fit()
print(model_goog_fit.summary())
#no autocorrelation
#non-significant AR terms
#significant MA term


### Third run
#### Set parameters
p = 0
q = 1


#### Fit model
model_goog = ARIMA(df_ac_log_diff['goog_adj_close'], order=(p, d, q))
model_goog_fit = model_goog.fit()
print(model_goog_fit.summary())
#no autocorrelation
#significant MA term


## Plot residuals
#### Assess normality and heteroscedasticity
resid_goog = pd.DataFrame(model_goog_fit.resid)

fig, ax = plt.subplots(1, 2)

ax[0].scatter(model_goog_fit.fittedvalues, resid_goog)
ax[0].axhline(0, color='red', linestyle='--')
# ax[0].set_xlim(0, 0.0012) #showws symmetrical distribution with low outliers
ax[0].set_xlabel('Fitted Values')
ax[0].set_ylabel('Residuals')
ax[0].set_title('Residuals versus fitted values \n(Heteroscedasticity)')

resid_msft.plot(ax=ax[1], kind='kde')
ax[1].set_xlabel('Residuals')
ax[1].set_ylabel('Density')
ax[1].set_title("Density of residuals \n(Normality)")
ax[1].get_legend().remove()

plt.tight_layout()

plot_goog_var_norm = plt.gcf()

plt.show()
plt.close()
#strong evidence of homoskedasticity and normality of residuals


#### Assess autocorrelation and partial autocorrelation
fig, ax = plt.subplots(1, 2)

plot_pacf(resid_goog, ax=ax[0], title="")
ax[0].set_ylabel("Partial autocorrelation")

plot_acf(resid_goog, ax=ax[1], title="")
ax[1].set_ylabel("Autocorrelation")

fig.supxlabel("Lag")

plt.tight_layout()

plot_goog_resid_pacf_acf = plt.gcf()

plt.show()
plt.close()
#no concerns



## Final sets of parameters-------------------------
#aapl: 0, 1
#msft: 2, 1
#amzn: 1, 1
#goog: 0, 1



# Write Data to File================================================================================
#combine models into dictionary
dict_models = {
  'aapl': model_aapl_fit,
  'msft': model_msft_fit,
  'amzn': model_amzn_fit,
  'goog': model_amzn_fit
}

#save file
# afile = open('fitted_models.pkl', 'wb')
# pickle.dump(dict_models, afile)
# afile.close()
