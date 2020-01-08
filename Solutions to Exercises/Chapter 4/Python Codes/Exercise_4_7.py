#%%
"""
Created on Wed Oct 16 2019
Exercise 4.7- Breeden-Litzenberger framework
@author: Irfan Ilgin & Lech A. Grzelak
"""
import numpy as np
import scipy.stats as st
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):

    # Black-Scholes call option price
    d1    = (np.log(S_0 / float(K)) + (r + 0.5 * np.power(sigma,2.0)) * tau) / float(sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if str(CP).lower()=="c" or str(CP).lower()=="1":
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif str(CP).lower()=="p" or str(CP).lower()=="-1":
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

def implied_vol(K):
    if K > 3:
        return implied_vol(3)
    else:
        iv = 0.510 - 0.591 * K + 0.376 * (K ** 2) - 0.105 * (K ** 3) + 0.011 * (K ** 4)
        return iv

def breeden_litzenberger(S, T, r, t, payoff_func):
    VpdH = lambda x: second_derivative(payoff_func, x) * BS_Call_Option_Price("p",S,x,implied_vol(x),T,r)#european_put(implied_vol(x), S, x, T, r, t)
    VcdH = lambda x: second_derivative(payoff_func, x) * BS_Call_Option_Price("c",S,x,implied_vol(x),T,r)#european_call(implied_vol(x), S, x, T, r, t)
    value = payoff_func(S) + quad(VpdH, 0, S)[0] + quad(VcdH, S, np.inf)[0]
    return value

def second_derivative(func, x, eps=1e-1):
    y_back = func(x - eps)
    y_forward = func(x + eps)
    y = func(x)
    df2_dx2 = (y_forward + y_back - 2 * y) / (eps ** 2)
    return df2_dx2

# Initial parameters and market quotes
V_market = 0.2;       # Market call option price
K        = 60.0;      # Strike price
tau      = 0.274;     # Time-to-maturity
r        = 0.05;      # Interest rate
S_0      = 50;        # Today's stock price
sigmaInit= 0.1;       # Initial implied volatility
CP       ="c"         # C is call and P is put

payoff1 = lambda S: max(S**2 - 1.2 * S, 0)
payoff2 = lambda S: max(S - 1.5, 0)
payoff3 = lambda S: max(1.7 - S, 0) + max(S - 1.4, 0)
payoff4 = lambda S: max(4 - S ** 3, 0) + max(S - 2, 0)

v1 = breeden_litzenberger(S=1, T=4, r=0, t=0, payoff_func=payoff1)
v2 = breeden_litzenberger(S=1, T=4, r=0, t=0, payoff_func=payoff2)
v3 = breeden_litzenberger(S=1, T=4, r=0, t=0, payoff_func=payoff3)
v4 = breeden_litzenberger(S=1, T=4, r=0, t=0, payoff_func=payoff4)

print("value1 = {0},\n value2 = {1},\n value3 = {2},\n value4 = {3}".format( v1, v2, v3, v4))

# To check the result we can calculate option2 and option3 using Black Scholes.
bs_value2 = BS_Call_Option_Price("c", S_0=1, K=1.5,sigma=implied_vol(1.5), tau=4, r=0.0)
bs_value3 = BS_Call_Option_Price("c", S_0=1, K=1.4,sigma=implied_vol(1.4), tau=4, r=0)\
             + BS_Call_Option_Price("p", S_0=1, K=1.7, sigma=implied_vol(1.7), tau=4, r=0)
             
# The value of the payoff2 and payoff3 can be determined withouth using the BL framework
print("value2 = {0} and value3 = {1}".format(bs_value2, bs_value3))