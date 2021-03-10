#%%
"""
Created on Nov 09 2019
Usage of the Cholesky decomposition in option pricing
@author: Lech A. Grzelak 
"""
import numpy as np
import matplotlib.pyplot as plt
import enum 
import scipy.stats as st

# set i= imaginary number
i   = np.complex(0.0,1.0)

# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def GeneratePathsEuler(NoOfPaths,NoOfSteps,T,S_0,rho):    
      
    W1 = np.zeros([NoOfPaths, NoOfSteps+1])
    W2 = np.zeros([NoOfPaths, NoOfSteps+1])
    W3 = np.zeros([NoOfPaths, NoOfSteps+1])
    
    S      = np.zeros([NoOfPaths, NoOfSteps+1])
    S[:,0] = S_0
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)   
    
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        Z = np.random.multivariate_normal([0,0,0],[[1,rho,rho],[rho,1,rho],[rho,rho,1]],NoOfPaths)
        Z1 = Z[:,0]
        Z2 = Z[:,1]
        Z3 = Z[:,2]
        if NoOfPaths > 1:
            Z1 = (Z1 - np.mean(Z1)) / np.std(Z1)
            Z2 = (Z2 - np.mean(Z2)) / np.std(Z2)
            Z3 = (Z3 - np.mean(Z3)) / np.std(Z3)
                
        W1[:,i+1] = W1[:,i] + np.power(dt, 0.5)*Z1
        W2[:,i+1] = W2[:,i] + np.power(dt, 0.5)*Z2
        W3[:,i+1] = W3[:,i] + np.power(dt, 0.5)*Z3
        
        S[:,i+1] = S[:,i] + 3.0/2.0*S[:,i]*dt + S[:,i]*((W1[:,i+1]-W1[:,i]) + (W2[:,i+1]-W2[:,i]) +(W3[:,i+1]-W3[:,i]))
        time[i+1] = time[i] + dt
        
    #Compute exponent
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
    NoOfPaths = 50000
    NoOfSteps = 200
    T   = .5
    S_0 = 1.0
    rho = 0.2
    r   = 0.0
    
    K= np.linspace(0.01,S_0*10.0,30)
    
    # Monte Carlo simulation
    np.random.seed(2)
    paths = GeneratePathsEuler(NoOfPaths,NoOfSteps,T,S_0,rho)
    S = paths["S"]
    
    # Generate S(T) given the solution in the solutions
    alpha = 1.0 + 2.0*rho
    beta  = (1.0+rho-2.0*rho**2.0)/np.sqrt(1.0-rho**2.0)
    gamma = np.sqrt(1.0 - 2.0 *rho**2.0)
    mu    = np.log(S_0) + (3.0/2.0 -0.5*(alpha**2.0+beta**2.0+gamma**2.0))*T
    sigma = np.sqrt(T*(alpha**2.0+beta**2.0+gamma**2.0)) 
    X     = mu + sigma* np.random.normal(0.0,1.0,[NoOfPaths,1])
    S2    = np.exp(X)
    
    pricesMC =[]
    pricesMCFactorized =[]
    for k in K:
        priceEuler = EUOptionPriceFromMCPaths(OptionType.CALL,S[:,-1],k,T,r)
        priceEulerFactorized = EUOptionPriceFromMCPaths(OptionType.CALL,S2[:,-1],k,T,r)
        
        pricesMC.append(priceEuler)
        pricesMCFactorized.append(priceEulerFactorized)
        
    
    plt.figure(1)
    plt.plot(K,pricesMC,'k')  
    plt.plot(K,pricesMCFactorized,'.r')  
    plt.grid()
    plt.legend(['Monte Carlo','Exact solution'])
    plt.xlabel('strike, K')
    plt.ylabel('option value')
    
mainCalculation()