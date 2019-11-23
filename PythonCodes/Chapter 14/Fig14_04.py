#%%
"""
Created on Jan 25 2019
Implied volatilities of the Displaced Diffusion model
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 

# Set i= imaginary number

i   = np.complex(0.0,1.0)

# Time step 

dt = 0.0001
 
# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
# Black-Scholes call option price

def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * tau) / (sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

# Implied volatility method

def ImpliedVolatilityBlack76_xxx(CP,marketPrice,K,T,S_0):
    func = lambda sigma: np.power(BS_Call_Option_Price(CP,S_0,K,sigma,T,0.0) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, 0.2, tol=1e-9)
    #impliedVol = optimize.brent(func, brack= (0.05, 2))
    return impliedVol

# Implied volatility method

def ImpliedVolatilityBlack76(CP,marketPrice,K,T,S_0):

    # To determine initial volatility we define a grid for sigma
    # and interpolate on the inverse function

    sigmaGrid = np.linspace(0.0,2.0,5000)
    optPriceGrid = BS_Call_Option_Price(CP,S_0,K,sigmaGrid,T,0.0)
    sigmaInitial = np.interp(marketPrice,optPriceGrid,sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))
    
    # Use already determined input for the local-search (final tuning)

    func = lambda sigma: np.power(BS_Call_Option_Price(CP,S_0,K,sigma,T,0.0) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-15)
    print("Final volatility = {0}".format(impliedVol))
    return impliedVol

def DisplacedDiffusionModel_CallPrice(K,P0T,beta,sigma,frwd,T):
     d1    = (np.log(frwd / (beta*K+(1.0-beta)*frwd)) + 0.5 * np.power(sigma*beta,2.0) * T) / (sigma * beta* np.sqrt(T))
     d2    = d1 - sigma * beta * np.sqrt(T)
     return P0T(T) * (frwd/beta * st.norm.cdf(d1) - (K + (1.0-beta)/beta*frwd) * st.norm.cdf(d2))

def mainCalculation():
    CP  = OptionType.CALL
        
    K = np.linspace(0.3,2.8,22)
    K = np.array(K).reshape([len(K),1])
    
    # We define a ZCB curve (obtained from the market)

    P0T = lambda T: np.exp(-0.05*T) 
    
    # DD model parameters

    beta  = 0.5
    sigma = 0.15
    
    # Forward rate

    frwd = 1.0
    
    # Maturity

    T = 2.0
           
    # Effect of sigma

    plt.figure(1)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    sigmaV = [0.1,0.2,0.3,0.4]
    legend = []
    for sigmaTemp in sigmaV:    
       callPrice = DisplacedDiffusionModel_CallPrice(K,P0T,beta,sigmaTemp,frwd,T)

       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           valCOSFrwd = callPrice[idx]/P0T(T)
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd,K[idx],T,frwd)
       plt.plot(K,IV*100.0)       
       legend.append('sigma={0}'.format(sigmaTemp))
       plt.ylim([0.0,60])
    plt.legend(legend)    
    
    # Effect of beta

    plt.figure(2)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    betaV = [-0.5, 0.0001, 0.5, 1.0]
    legend = []
    for betaTemp in betaV:    
       callPrice = DisplacedDiffusionModel_CallPrice(K,P0T,betaTemp,sigma,frwd,T)

       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           valCOSFrwd = callPrice[idx]/P0T(T)
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd,K[idx],T,frwd)
       plt.plot(K,IV*100.0)       
       legend.append('beta={0}'.format(betaTemp))
    plt.legend(legend)    
    
mainCalculation()
