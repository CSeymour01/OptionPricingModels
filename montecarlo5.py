# Import the necessary libraries
import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt

# Use the DataReader class to download data for the
# specified stock from Yahoo Finance
df = data.DataReader('SBUX', 'yahoo', '2021-10-01', '2022-10-01')

# this fills any missing values with the median
df['Close'] = df['Close'].fillna(df['Close'].median())

# Shift the 'Close' column by one position
df['Close_shift'] = df['Close'].shift(1)

# Calculate the ratio of each price to the previous price
df['Close_ratio'] = (df['Close'] / df['Close_shift'])

# Calculate the log returns of the stock's prices
df['Log_returns'] = df['Close_ratio'].shift(1).apply(np.log)

# Calculate the mean and standard deviation of the log returns
mean = df['Log_returns'].mean()
std = df['Log_returns'].std()

# Set the number of days to expiry
N_days = 5

# Set the number of simulation trials
N_runs = 1000000

# Set the price of the underlying security on a given day
Spot_Price = 105

# Set the strike price of the options contract
Strike_Price = 103

# Set the implied volatility
volatility = 0.15

# Generate random variables using the normal distribution
# and add a new axis to the array to make it two-dimensional
rets = np.expand_dims(np.random.normal(mean, std, N_runs), axis=1)*volatility/np.sqrt(252)
# Calculate the potential price paths using the random variables
traces = np.cumsum(rets, axis=1) * Spot_Price

# Extract the x and y coordinates from the traces array
x_coordinates = np.arange(N_days+1)
y_coordinates = traces[0,:]
y_coordinates = np.reshape(y_coordinates, (N_runs, N_days+1))

# Plot the first row of the traces array using the x and y coordinates
for i in range(N_runs):
    plt.plot(x_coordinates, y_coordinates[i,:])

plt.grid()
plt.xlabel('Days', fontsize=12)
plt.ylabel('Spot price', fontsize=12)
plt.show()
