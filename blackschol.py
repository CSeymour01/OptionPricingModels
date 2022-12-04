import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

np.random.seed(0)               #  Set a constant random seed for debugging
                                #  purposes

N_days = 5                     #  Number of days to simulate

S0 = 105.5                      #  Initial stock price
r = 0.0351 / 252.0                #  Risk-free rate (daily percent)
dt = 1.0                        #  Time step (days)
sigma = 0.225 / sqrt(252.0)      #  Volatility (daily percent)


#  Create a vector of normally distributed entries with each entry
#  corresponding to  one day
epsilon = np.random.normal(size=N_days)

ds_s = r * dt + sigma * sqrt(dt) * epsilon
#  Vector of daily percent change in the stock price

#  We need to build the diagonals of our sparse matrix
#  All of the main diagonal is equal to -1 expect the first entry which is
#  equal to one
ones = -np.ones( (N_days + 1) ); ones[0] = 1.;

#  Define our two diagonals,  the lower and main
d = [ds_s + 1, ones]

#  the K vector tells Python which of the vectors defined in 'd' go in
#  which diagonal.  Zero corresponds to the main, and -1 to the diag
#  immediately below.
K = [-1, 0]

#  Define the sparse matrix
M = scipy.sparse.diags(d, K, format='csc')

#  Define a column vector off all zeros expect for the first entry which is
#  our initial stock price.
p = np.zeros( (N_days + 1, 1) )
p[0] = S0

#  Solve the system  M * s = p for the vector s
s = scipy.sparse.linalg.spsolve(M, p)

#  Plot the results
t = np.arange(0, N_days + 1)
plt.plot(t, s, 'k')
plt.xlabel('Time (days)')
plt.ylabel('Stock Price ($)')
plt.grid(True)
plt.show()




