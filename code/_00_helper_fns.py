# This script contains helper functions for the stock price machine learning project


# Load Libraries====================================================================================
import pandas as pd
import math



# Define Functions==================================================================================
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




