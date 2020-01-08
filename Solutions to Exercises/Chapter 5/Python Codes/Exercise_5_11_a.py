#%%
"""
Created on Nov 09 2019
Merton model and the convergence for the exact solution
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 

# set i= imaginary number
i   = np.complex(0.0,1.0)

# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
# Black-Scholes Call option price
def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * tau) / (sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

def MertonCallPrice(CP,S0,K,r,tau,muJ,sigmaJ,sigma,xiP,n):
    X0  = np.log(S0)
    # term for E(exp(J)-1)
    helpExp = np.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0
             
    # Analytical representation for the Merton's option price
    muX     = lambda n: X0 + (r - xiP * helpExp - 0.5 * sigma * sigma) * tau + n * muJ
    sigmaX  = lambda n: np.sqrt(sigma * sigma + n * sigmaJ * sigmaJ / tau) 
    d1      = lambda n: (np.log(S0/K) + (r - xiP * helpExp - 0.5*sigma * sigma \
             + np.power(sigmaX(n),2.0)) * tau + n * muJ) / (sigmaX(n) * np.sqrt(tau))
    d2      = lambda n: d1(n) - sigmaX(n) * np.sqrt(tau)
    value_n = lambda n: np.exp(muX(n) + 0.5*np.power(sigmaX(n),2.0)*tau)\
            * st.norm.cdf(d1(n)) - K *st.norm.cdf(d2(n))
    # Option value calculation, it is infinite sum but we truncate at 20
    valueExact = value_n(0.0)
    kidx = range(1,n)
    for k in kidx:
        valueExact += np.power(xiP * tau, k)*value_n(k) / np.math.factorial(k)
    valueExact *= np.exp(-r*tau) * np.exp(-xiP * tau)
    
    if CP==OptionType.CALL:
        return valueExact
    elif CP==OptionType.PUT:
        return valueExact - S0 + K*np.exp(-r*tau)

def mainCalculation():
    CP  = OptionType.CALL
    S0  = 40
    r   = 0.06
    tau = 0.1
    
    K = 50 #np.linspace(1,3*S0,25)

    # Set 1
    sigma  = 0.2
    muJ    = -0.2
    sigmaJ = 0.2
    xiP    = 3.0

    # Set 2
    sigma  = 0.2
    muJ    = -0.2
    sigmaJ = 0.2
    xiP    = 8.0

    # Set 3
    sigma  = 0.2
    muJ    = -0.9
    sigmaJ = 0.45
    xiP    = .1
    
    leg = []
    plt.figure(1)    
    for n in range(1,7,1):
        valueExact = MertonCallPrice(CP,S0,K,r,tau,muJ,sigmaJ,sigma,xiP,n)
        plt.plot(K,valueExact)
        leg.append('n='+str(n))
        print('for ' +  leg[-1]+' value = '+str(valueExact))
    
    plt.xlabel("strike, K")
    plt.ylabel("Option Price")
    plt.grid()    
    plt.legend(leg)
    
    # Black Scholes price is given by
    value = BS_Call_Option_Price(CP,S0,K,sigma,tau,r)
    print('Black-Scholes price = {0}'.format(value))
    
mainCalculation()