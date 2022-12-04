from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.stats


def stock_monte_carlo(init_price, N_days, N_sims, r, sigma, reshape = True):
    #  Scale interest rates and volatility.  Define time-step.
    r = r / 252.0
    dt = 1.0
    sigma = sigma / sqrt(252.0)

    #  Calculate vector of normally distributed numbers and use it to
    #  calculate the daily percent change.
    epsilon = np.random.normal( size = (N_sims * N_days + N_sims - 1) )
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
#  columns.  This is dine by default but can be overridden by the user
    if reshape == True:
        s =  np.reshape(s, (N_sims, N_days+1))

    return s


def call_price(d1, d2, S, K, r, t):
    C = np.multiply(S, scipy.stats.norm.cdf(d1)) - \
    np.multiply(scipy.stats.norm.cdf(d2) * K, np.exp(-r * t))
    return C


def put_price(d1, d2, S, K, r, t):
    P = -np.multiply(S, scipy.stats.norm.cdf(-d1)) + \
    np.multiply(scipy.stats.norm.cdf(-d2) * K, np.exp(-r * t))
    return P


def d(S, K, r, sigma, t):
    d1 = np.multiply( 1. / sigma * np.divide(1., np.sqrt(t)),
        np.log(S/K) + (r + sigma**2 / 2.) * t  )
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2


np.random.seed(2)
np.seterr(divide = 'ignore')

N_days = 5                     #  Number of trading days
N_sims = 1000                   #  Number of simulations

#  Set up our parameters
dpy = 252.0                             #  Trading days per year
S0 = 105.5                               #  Initial stock price
K = 103.0                              #  Strike price
sigma = 0.225                            #  Implied volatility
r = 0.0351                                #  Risk free rate

#  Since we need te price of the call option at every day, we'll generate a
#  sequence days from N_days to zero
t_days = np.arange(N_days, 0, -1)
t_days = np.append(t_days, 0.)
t = t_days / dpy                        #  Convert days to yeears

dt = 1.                                 #  Time step in days

#  Run the Monte Carlo code to generate stock data
S = stock_monte_carlo(S0, N_days, N_sims, r, sigma)

#  Calculate the values of d1, d2, and the call Prices
d1, d2 = d(S, K, r, sigma, t)
P = put_price(d1, d2, S, K, r, t)

#  Plot put prices
# t = np.arange(0, N_days + 1)

for i in range(N_sims):
    plt.plot(t * 252, P[i,:], 'k', alpha=0.05)


plt.xlabel('Time to Expiration (days)')
plt.ylabel('Put Price ($)')
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()

plt.figure()
plt.hist(P[:,-1], bins = 25, edgecolor='k', linewidth=0.5 )
plt.xlabel('Put Price ($)')
plt.ylabel('Count')
# plt.yscale('log', nonposy='clip')
logbins=np.max(P)*(np.logspace(0, 1, num=1000) - 1)/5
hh,ee=np.histogram(P, density=True, bins=logbins)
plt.show()

#  What was the initial premium collected for the sale of the call?
initial_price = P[0,0]
print('Initial call price = ', initial_price)

#  P&L if held to expiration.  At expiration,  many losing trades are there?
losers = P[:, -1] > initial_price
print('Number of losing trades:  ', np.sum(losers))

winners = P[:, -1] <= initial_price
print('Number of winning trades:  ', np.sum(winners))

#  Sanity check by making sure the number of winners + losers equals the
#  total number of simulated trades
print('Sanity check: ', np.sum(winners) + np.sum(losers), ' == ', N_sims)
print('\n')


d1, d2 = d(100., K + initial_price, r, sigma, t[0])
print('Probability of profit from Black-Scholes:  ', 1 - scipy.stats.norm.cdf(d2))

print('Estimated probability of profit from Monte Carlo:  ', np.sum(winners) / float(N_sims))


win_ind = np.where( P[:, -1] <= initial_price )
wins = P[win_ind, -1]
print('Total profit from winners: ', np.sum( initial_price - wins ))

loss_ind = np.where( P[:, -1] > initial_price )
losses = P[loss_ind, -1]
print('Total losses: ', np.sum(losses - initial_price))


half_max = initial_price / 2.0
reached_half_max = P >= half_max
reached_half_max = np.sum(reached_half_max, axis=1)
print('Percentage of trades that reached 50% of max profit:  ', np.sum(reached_half_max > 0) / float(N_sims))

twice_max = initial_price * 2.
reached_twice_max = P <= twice_max
reached_twice_max = np.sum(reached_twice_max, axis=1)
print('Probability of loss at some point is 100% of the premium collected:  ', np.sum( reached_twice_max > 0 ) / float(N_sims))
