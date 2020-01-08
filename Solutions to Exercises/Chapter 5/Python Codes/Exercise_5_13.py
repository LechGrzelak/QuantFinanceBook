#%%
"""
Created on Nov 09 2019
Pricing of an option where volatility is jumpy
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as stFunc
import enum 
import scipy.stats as st

# set i= imaginary number
i   = np.complex(0.0,1.0)

# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
def GeneratePathsGBMwithJump(NoOfPaths,NoOfSteps,T,r,S_0,Jval,Jprob):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
   
    # Euler Approximation
    S = np.zeros([NoOfPaths, NoOfSteps+1])
    S[:,0] =S_0
    
    time = np.zeros([NoOfSteps+1])
    
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        J = np.random.choice(Jval, NoOfPaths, p=Jprob)
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        S[:,i+1] = S[:,i] + r * S[:,i]* dt + J * S[:,i] * (W[:,i+1] - W[:,i])
        time[i+1] = time[i] +dt
        
    paths = {"time":time,"S":S}
    return paths

def EUOptionPriceFromMCPaths(CP,S,K,T,r):
    # S is a vector of Monte Carlo samples at T
    if CP == OptionType.CALL:
        return np.exp(-r*T)*np.mean(np.maximum(S-K,0.0))
    elif CP == OptionType.PUT:
        return np.exp(-r*T)*np.mean(np.maximum(K-S,0.0))

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

def mainCalculation():
    NoOfPaths = 10000
    NoOfSteps = 1000
    T = 2.0
    r = 0.05
    S_0 = 1.0
    Jval = [0.1, 0.2]
    Jprob = [1.0/3.0, 2.0/3.0]
    K= np.linspace(0.01,3.0,30)
    
    # Monte Carlo simulation
    paths = GeneratePathsGBMwithJump(NoOfPaths,NoOfSteps,T,r,S_0,Jval,Jprob)
    S = paths["S"]
    
    pricesMC =[]
    for k in K:
        priceEuler = EUOptionPriceFromMCPaths(OptionType.CALL,S[:,-1],k,T,r)
        pricesMC.append(priceEuler)
        
    plt.figure(1)
    plt.plot(K,pricesMC,'k')
    
    # Exact based on analytical expression
    priceExact = np.zeros([len(K)])
    for i in range(0,len(Jval)):
        sigma = Jval[i]
        priceExact = priceExact + BS_Call_Option_Price(OptionType.CALL,S_0,K,sigma,T,r)*Jprob[i]
    
    plt.plot(K,priceExact,'--r')
    plt.grid()
    plt.legend(['Monte Carlo','Exact'])
    plt.xlabel('strike, K')
    plt.ylabel('option value')
mainCalculation()