import numpy as np
import pylab as plt
import scipy.stats as stats
import pandas as pd
from pandas_datareader import data


### Monte Carlo Starts here
# number of days to expiry
N_days = 5

# number of simulation trials
N_runs = 100000

# the price of the underlying security on a given day
Spot_Price = 105

# the strike price of the options contract
Strike_Price = 103

# the implied volatility
volatility = 0.2

# set a random seed for debugging purposes
np.random.seed(22)
# generate random variables (spot prices) and use the volatility to create realistic outcomes
# rets = np.random.randn(N_runs, N_days+1)*volatility/np.sqrt(252)
# rets.shape

# Use the DataReader class to download data for the
# specified stock from Yahoo Finance
df = data.DataReader('SBUX', 'yahoo', '2021-10-01', '2022-10-01')

# this fills any missing values with the median
df['Close'] = df['Close'].fillna(df['Close'].median())

# Calculate the log returns of the 'Close' column
# log_returns = df['Close'].pct_change().apply(np.log)

# Shift the 'Close' column by one position
df['Close_shift'] = df['Close'].shift(1)

# Calculate the ratio of each price to the previous price
df['Close_ratio'] = df['Close'] / df['Close_shift']

# Calculate the log returns of the stock's prices
df['Log_returns'] = df['Close_ratio'].apply(np.log)

# Calculate the mean and standard deviation of the log returns
mean = df['Log_returns'].mean()
std = df['Log_returns'].std()
# Calculate the mean and standard deviation of the log returns
# mean = log_returns.mean()
# std = log_returns.std()

# Print the results
print("Mean: ", mean)
print("Standard deviation: ", std)

# rets = np.random.lognormal(mean, std, (N_runs, N_days))*volatility/np.sqrt(252)
# Generate random variables using the normal distribution
# and add a new axis to the array to make it two-dimensional
rets = np.expand_dims(np.random.normal(mean, std, N_runs), axis=1)*volatility/np.sqrt(252)

# Calculate the potential price paths using the random variables
traces = np.cumprod(1+rets, axis=1) * Spot_Price
for i in traces[:100,:]:
    plt.plot(i)


mediantrace = np.median(traces)
meantrace = np.mean(traces)


# line graph of price path simulations
plt.grid()
plt.xlabel('days', fontsize=12)
plt.ylabel('Spot price', fontsize=12)
plt.show()


# histogram of spot prices
plt.hist(traces[:,-1], bins=100);

plt.axvline(mediantrace, color='r', linestyle='solid', linewidth=1)
plt.axvline(meantrace, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(meantrace*1.05, max_ylim*0.9, 'Mean: {:.2f}'.format(meantrace))
plt.text(mediantrace*0.9, max_ylim*0.9, 'Median: {:.2f}'.format(mediantrace))

print(mediantrace)
print(meantrace)

plt.title('Distribution of final prices')
plt.xlabel('Final prices', fontsize=12)
plt.ylabel('counts')
plt.show()


