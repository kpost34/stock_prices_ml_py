# This script contains helper functions for the stock price machine learning project


# Load Libraries====================================================================================
import pandas as pd
import math



# Define Functions==================================================================================
## Calculate mean absolute error (MAE)
def calc_mae(df, stock, places=2):
  n = len(df)
  y = stock + '_adj_close'
  yhat = stock + '_mean'
  
  s_diff_abs = abs(df[y] - df[yhat])
  mean_absolute_error = (1/n) * sum(s_diff_abs)
  mae = round(mean_absolute_error, places)
  return mae



## Calculate root mean squared error (RMSE)
def calc_rmse(df, stock, places=2):
  n = len(df)
  y = stock + '_adj_close'
  yhat = stock + '_mean'
  
  s_diff_sq = (df[y] - df[yhat])**2
  root_mean_square_error = math.sqrt((1/n) * sum(s_diff_sq))
  rmse = round(root_mean_square_error, places)
  return rmse




