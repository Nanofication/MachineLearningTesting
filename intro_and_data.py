import pandas as pd
import quandl
import math

# Adj. Close is most likely feature. Not exactly a label
# Pattern recognition for stock prices
# For regression, simplify your data as much as possible

# Stock Price example

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#Percent volatility (High - Low) Percent:
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0

# Percent Change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close' # Just a variable
df.fillna(-9999, inplace=True) #fillna = means not available You cannot have not existing data
# All data should be there. -99999 will be treated as an outlier

forecast_out= int(math.ceil(0.01 * len(df))) # Math.ceil rounds everything up to nearest whole
# Number of days out to predict. We're using data from 10 days ago to predict today

df['label'] = df[forecast_col].shift(-forecast_out) # We are shifting the column negatively. It will be shifted up
# So each row's label column will be the adjusted close 10 days in the future

# Label column, time into the future
df.dropna(inplace=True)
print(df.head())

# Features are attributes that may cause adjusted close prices in 0.01% days to change
