# This script contains helper functions for the stock price machine learning project


# Load Libraries====================================================================================
import pandas as pd



# Define Functions==================================================================================
def calc_mae(df, y, yhat):
  n = len(df)
  
  s_diff_abs = abs(df[y] - df[yhat])
  mae = (1/n) * sum(s_diff_abs)
  return mae
