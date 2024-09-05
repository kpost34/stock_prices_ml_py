# This script contains helper functions for the stock price machine learning project


# Load Libraries====================================================================================
import pandas as pd
import numpy as np
import math
from great_tables import GT



# Calculation Functions=============================================================================
## Calculate mean absolute error (MAE)
def calc_mae(df, stock, places=2):
  #create objs
  n = len(df)
  y = stock + '_adj_close'
  yhat = stock + '_mean'
  
  #calculate mae
  ser_diff_abs = abs(df[y] - df[yhat])
  mae_raw = (1/n) * sum(ser_diff_abs)
  mae = round(mae_raw, places)
  
  #calculate mean
  mean_raw = sum(df[y])/n
  mean = round(mean_raw, places)
  
  #calculate mae as percentage of mean
  pct = round((mae_raw/mean_raw) * 100, places)
  
  #combine into DF
  df_mae_mean_pct = pd.DataFrame({'mae': [mae], 'mean': [mean], 'mae_as_pct': [pct]}, 
                                 index=[stock])
                                 
  return df_mae_mean_pct



## Calculate root mean squared error (RMSE)
def calc_rmse(df, stock, places=2):
  #create objs
  n = len(df)
  y = stock + '_adj_close'
  yhat = stock + '_mean'
  
  #calculate rmse
  ser_diff_sq = (df[y] - df[yhat])**2
  rmse_raw = math.sqrt((1/n) * sum(ser_diff_sq))
  rmse = round(rmse_raw, places)
  
  #calculate mean
  mean_raw = sum(df[y])/n
  mean = round(mean_raw, places)
  
  #calculate rmse as percentage of mean
  pct = round((rmse_raw/mean_raw) * 100, places)
  
  #combine into DF
  df_rmse_mean_pct = pd.DataFrame({'rmse': [rmse], 'mean': [mean], 'rmse_as_pct': [pct]}, 
                                  index=[stock])
  
  return df_rmse_mean_pct



# Extraction Functions===============================================================================
## Extract model metrics and Ljung-Box test results
def extract_arima_info(model):
  
  #run Ljung-Box test (for autocorrelation)
  np_lb_results = model.test_serial_correlation(method='ljungbox', lags=10)
  lb_stat = np_lb_results[0, 0, 0]
  lb_p = np_lb_results[0, 1, 0]

  #generate DF of model summary
  df_summary = pd.DataFrame({
    "AIC": model.aic,
    "BIC": model.bic,
    "Log-Likelihood": model.llf,
    'Ljung-Box Statistic': lb_stat,
    'Ljung-Box p-value': lb_p
  }, index=[0]).round(3)

  return df_summary
  
  
## Extract ARIMA parameters
def extract_arima_params(model):
  df_params = pd.DataFrame({
    'Coef': model.params,
    'Std Err': model.bse,
    'z': model.params / model.bse,
    'P>|z|': model.pvalues,
    'CI Lower': model.conf_int()[0],
    'CI Upper': model.conf_int()[1]
  }).round(3).reset_index(names='parameter')
  
  return df_params
  

# Formatting Functions==============================================================================
## Format summary stats tables
def format_number_gt(df, col_range, prefix):
  r = range(col_range[0], col_range[1])
  
  summ_stats = df.iloc[:, np.r_[0, r]]
  summ_stats.columns = summ_stats.columns.str.replace(prefix, '')
  gt_summ_stats = GT(summ_stats)
  gt_summ_stats = gt_summ_stats.fmt_number(columns='volume', n_sigfig=4)
  return gt_summ_stats



