# This script does initial data cleaning/wrangling and performs EDA


# Load Libraries, Set Options, and Change WD========================================================
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


## Options 
# pd.options.display.max_columns = 20
# pd.options.display.max_rows = 20
# pd.options.display.max_colwidth = 80
# np.set_printoptions(precision=4, suppress=True)



# Source Functions and Data=========================================================================
## Functions
# from _00_helper_fns import make_grouped_barplot


## Data
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


### Moving average
#### Calculate moving average
df_ac = df[t_adj_close]


print("appl", "%s_day_MA" %(str(10)), sep="_")
pd.DataFrame.rolling(df_ac['aapl_adj_close'],10).mean()


# Plot out several moving averages
#copy df
df_ma = df_ac.copy()

t_adj_close #list of adj close names from before

ma_day = [10, 30, 50] #ma intervals

#create adj closing prices
for col in t_adj_close:
  for ma in ma_day:
    stock = re.sub("_.+$", "", col)
    col_name = stock + "_" + str(ma) + "_day_MA"
    df_ma[col_name]=pd.DataFrame.rolling(df_ma[col],ma).mean()


#### Plot moving average
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


fig, axes = plt.subplots(2, 2)

axes[0, 0].plot(df_ma.index, df_ma.aapl_adj_close, label="Daily adj closing price", linestyle='solid')
axes[0, 1].plot(df_ma.index, df_ma.msft_adj_close, label="Daily adj closing price", linestyle='solid')
axes[1, 0].plot(df_ma.index, df_ma.amzn_adj_close, label="Daily adj closing price", linestyle='solid')
axes[1, 1].plot(df_ma.index, df_ma.goog_adj_close, label="Daily adj closing price", linestyle='solid')

# plt.plot(df_ma.index, df_ma.aapl_10_day_MA, label="10-day", linestyle='dashed')
# plt.plot(df_ma.index, df_ma.aapl_30_day_MA, label="30-day", linestyle='dotted')
# plt.plot(df_ma.index, df_ma.aapl_50_day_MA, label="50-day", linestyle='dashdot')

axes[0, 0].set_xticks(ticks=xlabs, labels=xlabs)
axes[0, 1].set_xticks(ticks=xlabs, labels=xlabs)
axes[1, 0].set_xticks(ticks=xlabs, labels=xlabs)
axes[1, 1].set_xticks(ticks=xlabs, labels=xlabs)

axes[0, 0].set_title('aapl')
axes[0, 1].set_title('msft')
axes[1, 0].set_title('amzn')
axes[1, 1].set_title('goog')

fig.suptitle("Stock adjusted closing prices with moving averages from 2020-2022")
fig.supxlabel('Date')
fig.supylabel('Adjusted closing price ($)')

fig.legend(labels, loc='lower right', bbox_to_anchor=(1,-0.1))
plt.tight_layout()
plt.show()
plt.close()





# First create some toy data:
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# Create just a figure and only one subplot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

# Create four polar Axes and access them through the returned array
fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

# Share a X axis with each column of subplots
plt.subplots(2, 2, sharex='col')

# Share a Y axis with each row of subplots
plt.subplots(2, 2, sharey='row')

# Share both X and Y axes with all subplots
plt.subplots(2, 2, sharex='all', sharey='all')

# Note that this is the same as
plt.subplots(2, 2, sharex=True, sharey=True)

# Create figure number 10 with a single subplot
# and clears it if it already exists.
fig, ax = plt.subplots(num=10, clear=True)



### Create boxplots of adjusted stock prices



### Autocorrelation



### Risk and return
#return = % change in closing price of the stock over one year
#risk = standard deviation of those returns
#calcs can be done for shorter time periods: daily, weekly, or even monthly

#1) calculate daily percent change in closing prices for each stock 
#2) mean of #1 values (returns) & multiplied by 252 (= # of trading days in a year) = avg annual returns
#3) annual risk = std dev multiplied by the square root of the number of trading days in a year


## Correlation of returns
#calculate correlations of the returns calculated in previous section from the different stocks



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











