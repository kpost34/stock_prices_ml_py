# **Tech Stock Forecasting Project**
This project illustrates fitting time-series stock price data of four major technology companies with ARIMA models for forecasting.

## Summary
The idea for this project was inspired by a conversation with ChatGPT. The training and test data, which include daily stock prices (i.e., open, low, high, close, adjusted close) and trading volumes, come from the yfinance package and cover January 2016 through January 2019 with the last month used in forecasting and diagnostics. This project utilized four technology stocks: Apple (*aapl*), Microsoft (*msft*), Amazon (*amzn*), and Google (*goog*). The training data (2016-2018) were explored prior to fitting ARIMA models to differenced, log-transformed adjusted closing prices of each stock through a process that considered autocorrelation, parameter significance, heteroscedasticity, and normality of residuals. Final models were used in forecasting January 2019 adjusted closing prices and diagnostics were performed.


## Report
+ [Report: html format](https://kpost34.github.io/stock_prices_ml_py/) 

#### **Project Creator: Keith Post**
+ [Github Profile](https://github.com/kpost34) 
+ [LinkedIN Profile](https://www.linkedin.com/in/keith-post/)
+ [Email](mailto:keithhpost@gmail.com)
