#%%
"""
Created on Jan 30 2019
Implied volatilities of the CEV model
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
from mpl_toolkits import mplot3d
import enum 
import scipy.optimize as optimize

# Set i= imaginary number

i   = np.complex(0.0,1.0)

# Time step

dt = 0.0001
 
# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
# Black-Scholes call option price

def BS_Call_Put_Option_Price(CP,S_0,K,sigma,tau,r):
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

def ImpliedVolatilityBlack76(CP,marketPrice,K,T,S_0):

    # To determine initial volatility we define a grid for sigma
    # and interpolate on the inverse function

    sigmaGrid = np.linspace(0.0,2.0,5000)
    optPriceGrid = BS_Call_Put_Option_Price(CP,S_0,K,sigmaGrid,T,0.0)
    sigmaInitial = np.interp(marketPrice,optPriceGrid,sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))
    
    # Use already determined input for the local-search (final tuning)

    func = lambda sigma: np.power(BS_Call_Put_Option_Price(CP,S_0,K,sigma,T,0.0) \
                                          - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-15)
    print("Final volatility = {0}".format(impliedVol))
    return impliedVol

def DisplacedDiffusionModel_CallPrice(K,P0T,beta,sigma,frwd,T):
     d1    = (np.log(frwd / (beta*K+(1.0-beta)*frwd)) + 0.5 * np.power(sigma*beta,2.0) * T)\
                 / (sigma * beta* np.sqrt(T))
     d2    = d1 - sigma * beta * np.sqrt(T)
     return P0T(T) * (frwd/beta * st.norm.cdf(d1) - (K + (1.0-beta)/beta*frwd) * st.norm.cdf(d2))

def CEVModel_CallPutPrice(CP,K,frwd,P0T,T,beta,sigma):
    a = np.power(K,2.0*(1.0-beta))/(np.power(1.0-beta,2.0)*sigma*sigma*T)
    b = 1.0/(1.0-beta)
    c= np.power(frwd,2.0*(1.0-beta))/(np.power(1.0-beta,2.0)*sigma*sigma*T)
    
    if (beta<1.0 and beta>0.0):
        if CP == OptionType.CALL:
            return frwd*P0T(T)*(1.0 - st.ncx2.cdf(a,b+2.0,c)) - K *P0T(T) * \
                    st.ncx2.cdf(c,b,a)
        elif CP==OptionType.PUT:
            return K *P0T(T) * (1.0- st.ncx2.cdf(c,b,a)) - frwd * P0T(T) * \
                    st.ncx2.cdf(a,b+2.0,c)
    elif beta>1:
        if CP == OptionType.CALL:
            return frwd*P0T(T)*(1.0 - st.ncx2.cdf(c,-b,a)) - K *P0T(T) *\
                    st.ncx2.cdf(a,2.0-b,c)
        elif CP==OptionType.PUT:
            return K *P0T(T) * (1.0- st.ncx2.cdf(a,2.0-b,c)) - frwd * P0T(T)\
                    * st.ncx2.cdf(c,-b,a)
        
def mainCalculation():
    CP  = OptionType.CALL
        
    K = np.linspace(0.4,1.7,22)
    K = np.array(K).reshape([len(K),1])
    
    # We define a ZCB curve (obtained from the market)

    P0T = lambda T: np.exp(-0.05*T) 
    
    # CEV model parameters

    beta  = 0.25
    sigma = 0.2
    
    # Maturity

    T = 1.0
    
    # Forward rate

    frwd = 1.0/P0T(T)
           
    # Effect of sigma

    plt.figure(1)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    sigmaV = [0.1, 0.2, 0.3, 0.4]
    legend = []
    for sigmaTemp in sigmaV:    
       optPrice = CEVModel_CallPutPrice(CP,K,frwd,P0T,T,beta,sigmaTemp)

       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           valCOSFrwd = optPrice[idx]/P0T(T)
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
    betaV = [0.1,0.3,1.3, 1.5]
    legend = []
    for betaTemp in betaV:    
       optPrice = CEVModel_CallPutPrice(CP,K,frwd,P0T,T,betaTemp,sigma)

       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           valCOSFrwd = optPrice[idx]/P0T(T)
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd,K[idx],T,frwd)
       plt.plot(K,IV*100.0)
       legend.append('beta={0}'.format(betaTemp))
    plt.legend(legend)   
    
mainCalculation()
