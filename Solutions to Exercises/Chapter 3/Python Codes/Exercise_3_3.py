#%%
"""
Created on Wed Oct 16 2019
Exercise 3.3- plot payoff for a call opton
@author: Irfan Ilgin & Lech A. Grzelak
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import enum 

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def BS_Call_Put_Option_Price(CP,S_0,K,sigma,t,T,r):
    K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * (T-t)) / (sigma * np.sqrt(T-t))
    d2    = d1 - sigma * np.sqrt(T-t)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t))
     #   print(value)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t)) - st.norm.cdf(-d1)*S_0
      #  print(value)
    return value

# Input settings
S0    = 100
sigma = 0.15
r     = 0.03
T     = 2.0
K     = np.linspace(0.001,S0*3, 100)

# Black Scholes option price
callPrice = BS_Call_Put_Option_Price(OptionType.CALL,S0,K,sigma,0.0,T,r)

# Payoff function, which is a simply Black-Scholes model for time to maturity ->0
payoff = BS_Call_Put_Option_Price(OptionType.CALL,S0,K,sigma,T-0.00001,T,r)

plt.figure(1)
plt.grid()
plt.plot(K,callPrice)
plt.plot(K,payoff)
plt.legend(['option price','payoff'])
plt.xlabel('strike,K')
plt.ylabel('option price')
