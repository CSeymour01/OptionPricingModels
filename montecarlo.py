import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg


def stock_monte_carlo(init_price, N_days, N_sims, r, sigma):

    #  Scale interest rates and volatility.  Define time-step.
    r = r / 252.
    dt = 1.0
    sigma = sigma / sqrt(252.0)

    #  Calculate vector of normally distributed numbers and use it to
    #  calculate the daily percent change.
    epsilon = np.random.normal(size=(N_sims * N_days + N_sims - 1))
    ds_s = r * dt + sigma * sqrt(dt) * epsilon

    #  Step up matrix diagonals
    ones = -np.ones( (N_sims * N_days + N_sims) )
    ones[0:-1:N_days+1] = 1.

    ds_s[N_days:N_days * N_sims + N_sims:N_days+1] = -1
    d = [ds_s + 1, ones]
    K = [-1, 0]

    #  Solve the system of equations
    M = scipy.sparse.diags(d, K, format = 'csc')
    p = np.zeros( (N_sims * N_days + N_sims, 1) )
    p[0:-1:N_days+1] = init_price
    s = scipy.sparse.linalg.spsolve(M, p)

    #  Reshape the column vector so the function returns a matrix where
    #  each row is a single simulation with each day corresponding the the
    #  columns
    return np.reshape(s, (N_sims, N_days+1))


num_sims = 1000
N = 5

r = 0.0351
sigma = 0.225
S0 = 105.5

s = stock_monte_carlo(S0, N, num_sims,  r, sigma)

t = np.arange(0, N + 1)
for i in range(num_sims):
    plt.plot(t, s[i,:], 'k', alpha=0.05)

plt.xlabel('Time (days)')
plt.ylabel('Stock Price ($)')
plt.grid(True)
plt.show()


plt.hist(s[:,-1], bins = 50, edgecolor = 'k', linewidth = 0.5)
plt.xlabel('Stock Price ($)')
plt.ylabel('Count')
plt.show()
