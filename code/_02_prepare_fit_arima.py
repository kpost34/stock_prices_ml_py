# This script scales the target variable and fits a simple forecasting model


# Load Libraries and Data Import====================================================================
## Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import pickle
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
# from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy import stats
import yfinance as yf


## Data import
#change wd
os.chdir(str(Path.cwd()) + '/data') #change wd
Path.cwd()

#import training data
file = open('data_initial_clean.pkl', 'rb')
df0 = pickle.load(file)

#import test data
df_aapl_test = yf.download('AAPL', start='2019-01-01', end='2020-01-01')[['Adj Close']].rename(columns={"Adj Close": "aapl_adj_close"})
df_msft_test = yf.download('MSFT', start='2019-01-01', end='2020-01-01')[['Adj Close']].rename(columns={"Adj Close": "aapl_adj_close"})
df_goog_test = yf.download('GOOG', start='2019-01-01', end='2020-01-01')[['Adj Close']].rename(columns={"Adj Close": "aapl_adj_close"})
df_amzn_test = yf.download('AMZN', start='2019-01-01', end='2020-01-01')[['Adj Close']].rename(columns={"Adj Close": "aapl_adj_close"})


# Data Transformation===============================================================================
## Log-transform the training data
t_adj_close = ['aapl_adj_close', 'msft_adj_close', 'amzn_adj_close', 'goog_adj_close']
df_ac = df0[t_adj_close]
df_ac_log = df_ac.apply(np.log)



# Stationarity Check and Differencing===============================================================
## Stationarity
df_ac_log.apply(adfuller).iloc[1]
#no p-values < 0.5, so no series is stationary


## Differencing
df_ac_log_diff = df_ac_log.diff(axis=0).dropna()
df_ac_log_diff.apply(adfuller).iloc[1]
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
#partial autocorrelation at lag k is the autocorrelaiton between x(t) and x(t-k) that is not 
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
ax[0].set_title('Residuals vs Fitted Values')

resid_aapl.plot(ax=ax[1], kind='kde')
ax[1].set_xlabel('Residuals')
ax[1].set_ylabel('Density')
ax[1].set_title("Density of Residuals")
ax[1].get_legend().remove()

plt.show()
plt.close()
#homoscedastic: despite fitted values outliers, residuals are consistent throughout range
#normally distributed--symmetrical and somewhat bell-shaped curve

#understand residuals more
model_aapl_fit.fittedvalues.sort_values(ascending=True).head(30)
#very small residuals (high negative values) are at start of data and may indicate some type of
  #different phase or a model adjustment


#### Assess autocorrelation and partial autocorrelation
plot_acf(resid_aapl)
plt.show()
plt.close()

plot_pacf(resid_aapl)
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
ax[0].set_title('Residuals vs Fitted Values')

resid_msft.plot(ax=ax[1], kind='kde')
ax[1].set_xlabel('Residuals')
ax[1].set_ylabel('Density')
ax[1].set_title("Density of Residuals")
ax[1].get_legend().remove()

plt.show()
plt.close()
#strong evidence of homoskedasticity and normality of residuals


#### Assess autocorrelation and partial autocorrelation
plot_acf(resid_msft)
plt.show()
plt.close()

plot_pacf(resid_msft)
plt.show()
plt.close()
#no conerns



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
ax[0].set_title('Residuals vs Fitted Values')

resid_msft.plot(ax=ax[1], kind='kde')
ax[1].set_xlabel('Residuals')
ax[1].set_ylabel('Density')
ax[1].set_title("Density of Residuals")
ax[1].get_legend().remove()

plt.show()
plt.close()
#strong evidence of homoskedasticity and normality of residuals


#### Assess autocorrelation and partial autocorrelation
plot_acf(resid_amzn)
plt.show()
plt.close()

plot_pacf(resid_amzn)
plt.show()
plt.close()
#no conerns


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
ax[0].set_title('Residuals vs Fitted Values')

resid_msft.plot(ax=ax[1], kind='kde')
ax[1].set_xlabel('Residuals')
ax[1].set_ylabel('Density')
ax[1].set_title("Density of Residuals")
ax[1].get_legend().remove()

plt.show()
plt.close()
#strong evidence of homoskedasticity and normality of residuals


#### Assess autocorrelation and partial autocorrelation
plot_acf(resid_goog)
plt.show()
plt.close()

plot_pacf(resid_goog)
plt.show()
plt.close()
#no conerns


## Final sets of parameters-------------------------
#aapl: 0, 1
#msft: 2, 1
#amzn: 1, 1
#goog: 0, 1



# Model Forecasting=================================================================================
## aapl--------------------
### Get forecasted values
forecast_aapl = model_aapl_fit.get_forecast(steps=21)

forecast_aapl_mean = forecast_aapl.predicted_mean

forecast_aapl_ci = forecast_aapl.conf_int()

forecast_aapl_sum = forecast_aapl.summary_frame()
forecast_aapl_sum.index = df_aapl_log.iloc[:21].index

forecast_aapl_vals = model_aapl_fit.fittedvalues.iloc[0:20]


### Undo differencing and log transform
#get last known log-transformed value
last_log_aapl_value = np.log(df0['aapl_adj_close'].iloc[-1])

#undo differencing by calculating actual forecasted log-transformed values
aapl_log_forecast = forecast_aapl_sum.cumsum() + last_log_aapl_value

#undo log transform
aapl_forecast_value = np.exp(aapl_log_forecast)


### Actual test values
df_aapl_test = yf.download('AAPL', start='2019-01-01', end='2020-01-01')[['Adj Close']].rename(columns={"Adj Close": "aapl_adj_close"})


### Combine actual and forecasted values
pd.concat([aapl_forecast_value, df_aapl_test], axis=1).iloc[0:21,]





























# Data Transformation===============================================================================
#differencing yielded non-normality and non-constant variances for all four stocks, so a
  #transformation is necessary

# Yeo-Johnson 
## Apply transformation
ac_diff_yj = power_transform(df_ac_diff,
                             method='yeo-johnson')
df_ac_diff_yj = pd.DataFrame(ac_diff_yj,
                             columns=t_stock)
df_ac_diff_yj.index = df_ac_diff.index
                             

## Fit model
#aapl
model_ar_yj_aapl = ARIMA(df_ac_diff_yj['aapl'], order=(p, d, q))
model_ar_yj_aapl_fit = model_ar_yj_aapl.fit()
print(model_ar_yj_aapl_fit.summary())
#negligible improvement in normality and constant variance



# Box-Cox 
## Apply transformation
### Find constant
constant = df_ac_diff.apply(min).min()*-1
constant = math.ceil(constant)      


### Add constant to data
df_ac_diff_c = df_ac_diff.add(constant)


### Apply Box-Cox transform
ac_diff_bc, lambda_ = stats.boxcox(df_ac_diff_c['aapl_adj_close'])
df_ac_diff_bc = pd.DataFrame(ac_diff_bc, index=df_ac_diff.index, columns=['aapl'])


## Fit model
model_ar_bc_aapl = ARIMA(df_ac_diff_bc['aapl'], order=(p, d, q))
model_ar_bc_aapl_fit = model_ar_bc_aapl.fit()
print(model_ar_bc_aapl_fit.summary())



# Standardization
## Apply tranformation
ac_diff_std = StandardScaler().fit_transform(df_ac_diff)
df_ac_diff_std = pd.DataFrame(ac_diff_std,
                              columns=t_stock)

## Fit model
#aapl
model_ar_std_aapl = ARIMA(df_ac_diff_std['aapl'], order=(p, d, q))
model_ar_std_aapl_fit = model_ar_std_aapl.fit()
print(model_ar_std_aapl_fit.summary())
#again, no improvement in normality or constant variance



# Log 
## Apply transformation
df_ac_diff_log = np.log(df_ac_diff_c['aapl_adj_close'])


## Fit model
#aapl
model_ar_log_aapl = ARIMA(df_ac_diff_log, order=(p, d, q))
model_ar_log_aapl_fit = model_ar_log_aapl.fit()
print(model_ar_log_aapl_fit.summary())


# Plot residuals to understand them in detail
resid_aapl = model_ar_aapl_fit.resid

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(resid_aapl)
plt.title('Residuals')
plt.subplot(122)
plt.acorr(resid_aapl, maxlags=40)
plt.title('ACF of Residuals')
plt.show()
plt.close()

# Breusch-Pagan test
bp_test = het_breuschpagan(resid_aapl, model_ar_aapl_fit.model.exog)
print("Breusch-Pagan test:", bp_test)

# Normality test
_, p_value = normal_ad(resid_aapl)
print("Shapiro-Wilk p-value:", p_value)



#given lack of improvement, double differencing applied



# Double Differencing===============================================================================
## Apply double differencing
df_ac_diff2 = df_ac_diff.diff(axis=0).dropna()
df_ac_diff2.apply(adfuller).iloc[1]
#all 0s, so all series are stationary



# Identify ARIMA Parameters (p, d,  q)==============================================================
## Autocorrelation (ACF)
fig, axes = plt.subplots(2, 2)

t_stock = ['aapl', 'msft', 'amzn', 'goog']
n = 0

for i in range(0, 2):
  for j in range(0, 2):
    col = t_adj_close[n]
    stock = t_stock[n]
    
    plot_acf(df_ac_diff2[col], ax=axes[i, j])
    axes[i, j].set_title(stock)
    
    n = n + 1

fig.subplots_adjust(top=1)
fig.suptitle("Autocorrelation of adjusted closing prices of \ntech stocks from 2020-2022")
fig.supxlabel('Lag')
fig.supylabel('Autocorrelation')
    
plt.tight_layout()
plt.show()
plt.close()
#q = 1 for all four stocks


## Partial Autocorrelation Function (PACF)
fig, axes = plt.subplots(2, 2)

t_stock = ['aapl', 'msft', 'amzn', 'goog']
n = 0

for i in range(0, 2):
  for j in range(0, 2):
    col = t_adj_close[n]
    stock = t_stock[n]
    
    plot_pacf(df_ac_diff2[col], ax=axes[i, j])
    axes[i, j].set_title(stock)
    
    n = n + 1

fig.subplots_adjust(top=1)
fig.suptitle("Partial autocorrelation of adjusted closing prices of \ntech stocks from 2020-2022")
fig.supxlabel('Lag')
fig.supylabel('Partial autocorrelation')
    
plt.tight_layout()
plt.show()
plt.close()
#p is the number of lags before spikes drop to zero/near-zero; this is 6

#therefore, p = 6, q = 1, and d = 2 for all stocks


# Fit ARIMA Model===================================================================================
p = 6
q = 1
d = 2

## aapl
model2_ar_aapl = ARIMA(df_ac_diff2['aapl_adj_close'], order=(p, d, q))
model2_ar_aapl_fit = model2_ar_aapl.fit()
print(model2_ar_aapl_fit.summary())

#same issues with non-normality and non-constant variance of residuals. Let's try a different model



# Fit SARIMAX model=================================================================================
p = 0
d = 1
q = 0
P = 1
D = 1
Q = 1
s = 252

model_sar_aapl = SARIMAX(df_ac_diff['aapl_adj_close'],
                         order=(p, d, q),
                         seasonal_order=(P, D, Q, s))
results = model_sar_aapl.fit()
print(results.summary())


                             


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



