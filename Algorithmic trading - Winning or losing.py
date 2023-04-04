#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Ignore printing all warnings
import warnings
warnings.filterwarnings('ignore')

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pyfolio as pf
import datetime as dt
import pandas_datareader.data as web
import os
import pyfolio as pf
import pandas_ta as ta

# print all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('max_columns', 30) 

# downloading historical necessary data for backtesting and analysis
_start = dt.date(2022,1,1)
_end = dt.date(2022,12,31)
ticker = 'UNH'
df = yf.download(ticker, start = _start, end = _end)


# calculating buy and hold strategy returns
df['bnh_returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
df.head(3)

###Strategy 1: Bollinger bands strategy
# creating bollinger band indicators
df['ma20'] = df['Adj Close'].rolling(window=20).mean()
df['std'] = df['Adj Close'].rolling(window=20).std()
df['upper_band'] = df['ma20'] + (2 * df['std'])
df['lower_band'] = df['ma20'] - (2 * df['std'])
#df.drop(['Open','High','Low'],axis=1,inplace=True,errors='ignore')
df.tail(5)

# BUY condition
df['bb_signal'] = np.where((df['Adj Close'] < df['lower_band']) &
                        (df['Adj Close'].shift(1) >= df['lower_band']),1,0)

# SELL condition
df['bb_signal'] = np.where( (df['Adj Close'] > df['upper_band']) &
                          (df['Adj Close'].shift(1) <= df['upper_band']),-1,df['bb_signal'])
# creating long and short positions 
df['bb_position'] = df['bb_signal'].replace(to_replace=0, method='ffill')

# shifting by 1, to account of close price return calculations
df['bb_position'] = df['bb_position'].shift(1)

# calculating stretegy returns
df['bb_strategy_returns'] = df['bnh_returns'] * (df['bb_position'])


##Strategy 2: Day-trading strategy
# BUY condition - Buy when the adj_close price is 1.9% higher than Open
df['dt_signal'] = np.where((df['Adj Close'] >= df['Open']*(1+0.019)),1,0)

# SELL condition - Sell when the adj_close price is 1.9% lower than Open
df['dt_signal'] = np.where( (df['Adj Close'] < df['Open']*(1-0.019)),-1,df['dt_signal'])

# creating long and short positions 
df['dt_position'] = df['dt_signal'].replace(to_replace=0, method='ffill')

# shifting by 1, to account of close price return calculations
df['dt_position'] = df['dt_position'].shift(1)

# calculating stretegy returns
df['dt_strategy_returns'] = df['bnh_returns'] * (df['dt_position'])



###Strategy 3: MACD strategy
macd = ta.macd(df['Adj Close'])

# BUY condition
df['macd_signal'] = np.where((macd['MACD_12_26_9'] < macd['MACDs_12_26_9']),1,0)
# SELL condition
df['macd_signal'] = np.where((macd['MACD_12_26_9'] > macd['MACDs_12_26_9']),-1,df['macd_signal'])
# creating long and short positions 
df['macd_position'] = df['macd_signal'].replace(to_replace=0, method='ffill')

# shifting by 1, to account of close price return calculations
df['macd_position'] = df['macd_position'].shift(1)

# calculating stretegy returns
df['macd_strategy_returns'] = df['bnh_returns'] * (df['macd_position'])

df.tail(5)


# comparing buy & hold strategy with 
#1) bollinger bands strategy returns
#2) Day-trading strategy returns
#3) MACD strategy returns
print("Buy and hold returns:",round(df['bnh_returns'].cumsum()[-1],3)*100,"%")
print("Bollinger bands strategy returns:",round(df['bb_strategy_returns'].cumsum()[-1],3)*100,"%")
print("Day-trading strategy returns:",round(df['dt_strategy_returns'].cumsum()[-1],3)*100,"%")
print("MACD strategy returns:",round(df['macd_strategy_returns'].cumsum()[-1],3)*100,"%")

# plotting strategy historical performance over time
df[['bnh_returns','bb_strategy_returns','dt_strategy_returns','macd_strategy_returns']] = df[['bnh_returns','bb_strategy_returns','dt_strategy_returns','macd_strategy_returns']].cumsum()
df[['bnh_returns','bb_strategy_returns','dt_strategy_returns','macd_strategy_returns']].plot(grid=True, figsize=(12, 8))
print('\n')
print("Bollinger Bands Strategy Returns: Cumulative returns, Rolling Sharpe Ratio, Underwater Plot")
print('')
pf.create_simple_tear_sheet(df['bb_strategy_returns'].diff())
print('\n')
print("Day-trading Strategy Returns: Cumulative returns, Rolling Sharpe Ratio, Underwater Plot")
print('')
pf.create_simple_tear_sheet(df['macd_strategy_returns'].diff())
print('\n')
print("MACD Strategy Returns: Cumulative returns, Rolling Sharpe Ratio, Underwater Plot")
print('')
pf.create_simple_tear_sheet(df['dt_strategy_returns'].diff())
