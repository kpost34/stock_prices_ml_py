# This script does initial data cleaning/wrangling and performs EDA


# Load Libraries====================================================================================
## Load libraries
import pandas as pd
import inflection
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import yfinance as yf
import re
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf
from pathlib import Path
import os



# Source Data=======================================================================================
df_aapl0 = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
df_msft0 = yf.download('MSFT', start='2020-01-01', end='2023-01-01')
df_amzn0 = yf.download('AMZN', start='2020-01-01', end='2023-01-01')
df_goog0 = yf.download('GOOG', start='2020-01-01', end='2023-01-01')



# Initial Wrangling=================================================================================
## Copy DFs
df_aapl = df_aapl0.copy()
df_msft = df_msft0.copy()
df_amzn = df_amzn0.copy()
df_goog = df_goog0.copy()


## Combine dataframes
### Extract and wrangle cols
cols_raw0 = df_aapl0.columns.map(inflection.underscore).tolist()
cols_raw = [re.sub(" ", "_", string) for string in cols_raw0]


### Add cols to each DF
df_aapl.columns = ["aapl_" + x for x in cols_raw if not str(x) == "nan"]
df_msft.columns = ["msft_" + x for x in cols_raw if not str(x) == "nan"]
df_amzn.columns = ["amzn_" + x for x in cols_raw if not str(x) == "nan"]
df_goog.columns = ["goog_" + x for x in cols_raw if not str(x) == "nan"]


### Combine DFs
df = pd.concat([df_aapl, df_msft, df_amzn, df_goog], axis=1)



# Data Checking=====================================================================================
## General info
df.info() 
#volume cols are int64, all other stock price cols are float64, and year and
  #month are int32


## Missingness
print(df.isnull().sum()) #no missing values


## Duplicates
print(df.duplicated().sum()) #no duplicate rows


## Outliers
t_adj_close = ['aapl_adj_close', 'msft_adj_close', 'amzn_adj_close', 'goog_adj_close']

s_q3 = df[t_adj_close].quantile(0.75)
s_q1 = df[t_adj_close].quantile(0.25)

s_iqr = s_q3 - s_q1
s_ub = s_q3 + 1.5 * s_iqr
s_lb = s_q1 - 1.5 * s_iqr

#Apple
df[(df['aapl_adj_close'] >= s_ub[s_ub.index=='aapl_adj_close'].item())|
     (df['aapl_adj_close'] <= s_lb[s_lb.index=='aapl_adj_close'].item())].aapl_adj_close
#7 values, all low and in March/April 2020

#MS
df[(df['msft_adj_close'] >= s_ub[s_ub.index=='msft_adj_close'].item())|
     (df['msft_adj_close'] <= s_lb[s_lb.index=='msft_adj_close'].item())].msft_adj_close
#none

#Amazon
df[(df['amzn_adj_close'] >= s_ub[s_ub.index=='amzn_adj_close'].item())|
     (df['amzn_adj_close'] <= s_lb[s_lb.index=='amzn_adj_close'].item())].amzn_adj_close
#none

#Google
df[(df['goog_adj_close'] >= s_ub[s_ub.index=='goog_adj_close'].item())|
     (df['goog_adj_close'] <= s_lb[s_lb.index=='goog_adj_close'].item())].goog_adj_close
#none



# Exploratory Data Analysis=========================================================================
## Plots
### Stock prices over time
xlabs = ["2020-01", "2020-07", "2021-01", "2021-07", "2022-01", "2022-07", "2023-01"]

plt.plot(df.index, df.msft_adj_close, label="Microsoft", color='darkred')
plt.plot(df.index, df.aapl_adj_close, label="Apple", color='skyblue')
plt.plot(df.index, df.amzn_adj_close, label="Amazon", color='purple')
plt.plot(df.index, df.goog_adj_close, label="Google", color='green')

plt.ylim(0, 400)
plt.xticks(ticks=xlabs, labels=xlabs)

plt.title("Tech stock adjusted closing prices 2020-2022")
plt.xlabel("Date")
plt.ylabel("Adjust closing price ($)")

plt.legend()

plt.show()
plt.close()
#msft: strong growth in 2020 and 2021 until peaks in late 2021/early 2022 before a decreasing
  #trend in 2022
#aapl: similar pattern as msft except less growth in absolute or relative terms then a period
  #of osciallation in 2022 before a decline to end the year
#amzn: growth in first half of 2020 then stability for rest of year and through 2021 before
  #a marked decline in 2022
#goog: steady growth in 2020 and 2021 before a slower but steady decline in 2022


### Normalized stock prices over time
#normalize data
df_ac = df[t_adj_close]

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_ac))
df_scaled.columns = t_adj_close
df_scaled.index = df.index
df_scaled

#plot data
plt.plot(df_scaled.index, df_scaled.msft_adj_close, label="Microsoft", color='darkred')
plt.plot(df_scaled.index, df_scaled.aapl_adj_close, label="Apple", color='skyblue')
plt.plot(df_scaled.index, df_scaled.amzn_adj_close, label="Amazon", color='purple')
plt.plot(df_scaled.index, df_scaled.goog_adj_close, label="Google", color='green')

plt.xticks(ticks=xlabs, labels=xlabs)

plt.title("Tech stock min-max scaled adjusted closing prices 2020-2022")
plt.xlabel("Date")
plt.ylabel("Adjust closing price ($)")

plt.legend()

plt.show()
plt.close()


 
### Volumes over time
plt.plot(df.index, df.aapl_volume, label="Apple", color='skyblue')
plt.plot(df.index, df.msft_volume, label="Microsoft", color='darkred')
plt.plot(df.index, df.amzn_volume, label="Amazon", color='purple')
plt.plot(df.index, df.goog_volume, label="Google", color='green')

plt.xticks(ticks=xlabs, labels=xlabs)

plt.title("Tech stock trading volumes 2020-2022")
plt.xlabel("Date")
plt.ylabel("Volume")

plt.legend()

plt.show()
plt.close()
#in order of descending volume: aapl, msft, goog, and amzn
#aapl markedly more volume in first half of 2020 compared to rest of 2020-2022
#the other three stocks had similar volumes over the study period
#msft had more volatility in volme compared to amzn and goog


### Moving average
#### Calculate moving average
#isolate adj closing prices
df_ac = df[t_adj_close]

#copy df
df_ma = df_ac.copy()

t_adj_close #list of adj close names from before

ma_day = [10, 50] #ma intervals
# ma_day = [10, 30, 50] #ma intervals

#create adj closing prices
for col in t_adj_close:
  for ma in ma_day:
    stock = re.sub("_.+$", "", col)
    col_name = stock + "_" + str(ma) + "_day_MA"
    df_ma[col_name]=pd.DataFrame.rolling(df_ma[col],ma).mean()


#### Plot moving average for Apple
df_ma.head()

plt.plot(df_ma.index, df_ma.aapl_adj_close, label="Daily adj closing price", linestyle='solid')
plt.plot(df_ma.index, df_ma.aapl_10_day_MA, label="10-day", linestyle='dashed')
plt.plot(df_ma.index, df_ma.aapl_30_day_MA, label="30-day", linestyle='dotted')
plt.plot(df_ma.index, df_ma.aapl_50_day_MA, label="50-day", linestyle='dashdot')

plt.xticks(ticks=xlabs, labels=xlabs)

plt.title("Apple stock adjusted closing prices with moving averages from 2020-2022")
plt.xlabel("Date")
plt.ylabel("Adjusted closing price ($)")

plt.legend()

plt.show()
plt.close()


#### Plot out several moving averages
fig, axes = plt.subplots(2, 2)
t_stock = ['aapl', 'msft', 'amzn', 'goog']
xlabs2 = ['2020-01', '2021-01', '2022-01', '2023-01']
n = 0

for i in range(0, 2):
  for j in range(0, 2):
    col = t_adj_close[n]
    stock = t_stock[n]
    col2 = stock + "_10_day_MA"
    col3 = stock + "_30_day_MA"
    col4 = stock + "_50_day_MA"
    
    axes[i, j].plot(df_ma.index, df_ma[col], label="Adj closing \nprice", linestyle='solid')
    axes[i, j].plot(df_ma.index, df_ma[col2], label="10-day MA", linestyle='dashed')
    # axes[i, j].plot(df_ma.index, df_ma[col3], label="30-day MA", linestyle='dotted')
    axes[i, j].plot(df_ma.index, df_ma[col4], label="50-day MA", linestyle='dashdot')

    axes[i, j].set_xticks(ticks=xlabs2, labels=xlabs2)
    axes[i, j].set_title(stock)
  
    n = n + 1
      
fig.suptitle("Adjusted closing prices with moving averages from 2020-2022")
fig.supxlabel('Date')
fig.supylabel('Adjusted closing price ($)')

axes.flatten()[-2].legend(loc='right', bbox_to_anchor=(3, 1), ncol=1)

fig.subplots_adjust(left=0.1, bottom=None, right=0.75, top=None, wspace=0.2, hspace=0.3)

plt.show()
plt.close()
#msft and goog: clear longer term patterns, which are captured by the 50-day MA plots
#aapl and amzn: the 50-day MA plots captured some of the volatility in the daily adj closing
  #prices


### Create boxplots of adjusted stock prices
df_ac_box = df_ac.copy()
df_ac_box.columns = t_stock
df_ac_box.plot(kind='box', ylim=(0, 400), xlabel="Stock", ylabel="Adjusted closing stock price ($)")

plt.show()
plt.close()
#msft adj stock prices are significantly greater than the other three
#only aapl contained outliers during this time period for these four stocks


### Risk and return
#return = % change in closing price of the stock over one year
#risk = standard deviation of those returns
#calcs can be done for shorter time periods: daily, weekly, or even monthly

df_ret = df_ac.copy()

#1) calculate daily percent change in closing prices for each stock
for i in range(4):
  col = t_adj_close[i]
  stock = t_stock[i]
  
  col_name = stock + "_return"
  df_ret[col_name] = df[col].pct_change().dropna().mul(100)
  
df_ret = df_ret.iloc[:, 4:] #retain return columns only
df_ret['year'] = pd.DatetimeIndex(df.index).year #extracts & adds year col

df_risk = df_ret.copy()

#2) mean of #1 values (returns) & multiplied by 252 (= # of trading days in a year) = avg annual returns
df_ret_yr = df_ret.groupby('year').mean() * 252 #avg annual returns
df_ret_yr.columns = t_stock

df_ret_long = pd.melt(df_ret_yr.reset_index(), var_name='stock', value_name='return', id_vars='year')


#3) annual risk = std dev multiplied by the square root of the number of trading days in a year
df_risk_yr = df_risk.groupby('year').std() * np.sqrt(252)
df_risk_yr.columns = t_stock

df_risk_long = pd.melt(df_risk_yr.reset_index(), var_name='stock', value_name='risk', id_vars='year')

#4) merge them together
df_rr = df_ret_long.merge(df_risk_long, how='inner', on=['year', 'stock'])

#5) plot annual risk vs return
#2020
df_rr_2020 = df_rr[df_rr['year']==2020]

plt.scatter(df_rr_2020["risk"], df_rr_2020["return"])
for i, row in df_rr_2020.iterrows():
    plt.annotate(row['stock'], (row['risk']+0.002, row['return']+0.01), fontsize=9, ha='right')

plt.show()
plt.close()


#iterate over each year
fig, axes = plt.subplots(2, 2)

n = 0
t_year = [2020, 2021, 2022]

when n <= 2:
  for i in [0, 1]:
    for j in [0, 1]:
      year = t_year[n]
      df_rr_yr = df_rr[df_rr['year']==year]
      
      axes[i,j].scatter(df_rr_yr["risk"], df_rr_yr["return"])
      
      for k, row in df_rr_yr.iterrows():
        axes[i,j].annotate(row['stock'], (row['risk']+0.002, row['return']+0.01), fontsize=9, ha='right')
      
      axes[i,j].set_title(year)
    
      n = n + 1
      
axes[1,1].remove()

fig.suptitle("Risk-return plots of four tech stocks from 2020-2022")
fig.supxlabel('Risk')
fig.supylabel('Return')

plt.tight_layout()
plt.show()
plt.close()
#order of decreasing return/risk
#2020: amzn, aapl (highest ret), msft, and goog
#2021: msft & goog highest ratio, followed by aapl and finally amzn
#2022: all negative returns


## Correlation of returns
#show returns correlations via heatmap/correlation matrix
df_ret_no_yr = df_ret.drop('year', axis=1)
df_ret_no_yr.columns = t_stock

corr_matrix = df_ret_no_yr.corr()

plt.figure(figsize=(8, 6))

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)

plt.title('Correlation of Tech Stock Returns')
plt.show()
plt.close()
#strong corrs (r > 0.8): aapl-msft, msft-goog
#med corrs (0.7 <= r <= 0.8): aapl-goog, msft-amzn
#weak corrs (r < 0.7): aapl-amzn, amzn-goog


## Calculate basic stats (mean, median, st dev)
### Overall
df.apply("describe")


### Add year and month to DF
df['year'] = pd.DatetimeIndex(df.index).year #extract year
df['month'] = pd.DatetimeIndex(df.index).month


### Summary stats by year
df.groupby('year').describe() #everything

#### By stock and year and subset of functions
fns = ['count', 'min', 'mean', 'median', 'max', 'std']

df[df.columns[df.columns.str.startswith(('aapl', 'year'))]].groupby('year').agg(fns)
df[df.columns[df.columns.str.startswith(('msft', 'year'))]].groupby('year').agg(fns)
df[df.columns[df.columns.str.startswith(('amzn', 'year'))]].groupby('year').agg(fns)
df[df.columns[df.columns.str.startswith(('goog', 'year'))]].groupby('year').agg(fns)


#### By stock and year and subset of cols and fns
cols_subset = ['low', 'high', 'adj_close', 'volume']

aapl_cols = ["aapl_" + x for x in cols_subset if not str(x) == "nan"] + ['year', 'month']
msft_cols = ["msft_" + x for x in cols_subset if not str(x) == "nan"] + ['year', 'month']
amzn_cols = ["amzn_" + x for x in cols_subset if not str(x) == "nan"] + ['year', 'month']
goog_cols = ["goog_" + x for x in cols_subset if not str(x) == "nan"] + ['year', 'month']

df[aapl_cols].groupby('year').agg(fns)
df[msft_cols].groupby('year').agg(fns)
df[amzn_cols].groupby('year').agg(fns)
df[goog_cols].groupby('year').agg(fns)


#### By stock, year, and month and subset of cols and fns
df[aapl_cols].groupby(['year', 'month']).agg(fns)
df[msft_cols].groupby(['year', 'month']).agg(fns)
df[amzn_cols].groupby(['year', 'month']).agg(fns)
df[goog_cols].groupby(['year', 'month']).agg(fns)



# Write Data to File================================================================================
#save in pickle format to retain data types

#change wd
os.chdir(str(Path.cwd()) + '/data') #change wd
Path.cwd() #returns new wd

#save file
# afile = open('data_initial_clean.pkl', 'wb')
# pickle.dump(df, afile)
# afile.close


