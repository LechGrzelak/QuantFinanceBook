#%%
"""
Created on 10 Mar 2019
Pathwise estimation for delta and vega for Asian option
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 


# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

# Black-Scholes call option price

def BS_Call_Put_Option_Price(CP,S_0,K,sigma,t,T,r):
    K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * (T-t)) / (sigma * np.sqrt(T-t))
    d2    = d1 - sigma * np.sqrt(T-t)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t))
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t)) - st.norm.cdf(-d1)*S_0
    return value

def BS_Delta(CP,S_0,K,sigma,t,T,r):
    K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * \
             (T-t)) / (sigma * np.sqrt(T-t))
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1)
    elif CP == OptionType.PUT:
       value = st.norm.cdf(d1)-1
    return value

def BS_Gamma(S_0,K,sigma,t,T,r):
    K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * \
             (T-t)) / (sigma * np.sqrt(T-t))
    return st.norm.pdf(d1) / (S_0 * sigma * np.sqrt(T-t))

def BS_Vega(S_0,K,sigma,t,T,r):
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * \
             (T-t)) / (sigma * np.sqrt(T-t))
    return S_0*st.norm.pdf(d1)*np.sqrt(T-t)

def GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
   
    # Approximation

    S = np.zeros([NoOfPaths, NoOfSteps+1])
    S[:,0] =S_0
    
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    X[:,0] =np.log(S_0)
      
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        X[:,i+1] = X[:,i] + (r -0.5*sigma**2.0)* dt + sigma * (W[:,i+1] - W[:,i])
        time[i+1] = time[i] +dt
        
    # Return S

    paths = {"time":time,"S":np.exp(X)}
    return paths

def EUOptionPriceFromMCPathsGeneralized(CP,S,K,T,r):

    # S is a vector of Monte Carlo samples at T

    result = np.zeros([len(K),1])
    if CP == OptionType.CALL:
        for (idx,k) in enumerate(K):
            result[idx] = np.exp(-r*T)*np.mean(np.maximum(S-k,0.0))
    elif CP == OptionType.PUT:
        for (idx,k) in enumerate(K):
            result[idx] = np.exp(-r*T)*np.mean(np.maximum(k-S,0.0))
    return result

def PathwiseDelta(S0,S,K,r,T):
    temp1 = S[:,-1]>K
    return np.exp(-r*T)*np.mean(S[:,-1]/S0*temp1)

def PathwiseVega(S0,S,sigma,K,r,T):
    temp1 = S[:,-1]>K
    temp2 = 1.0/sigma* S[:,-1]*(np.log(S[:,-1]/S0)-(r+0.5*sigma**2.0)*T)
    return np.exp(-r*T)*np.mean(temp1*temp2)

# Value of Asian option given stock paths

def AsianOption(S,time,Ti,r,T,cp,K,nPaths):
    A = np.zeros([nPaths])
    for ti in Ti:
        idx =(np.abs(time-ti)).argmin()
        A = A+ S[:,idx]
    A = A /len(Ti)    
        
    if cp ==OptionType.CALL:
        value = np.exp(-r*T)*np.mean(np.maximum(A-K,0.0))
    elif cp ==OptionType.PUT:
        value = np.exp(-r*T)*np.mean(np.maximum(K-A,0.0))
    asian={"value":value,"A":A}
    return asian

def PathwiseVegaAsian(S0,S,time,sigma,K,r,T,Ti,nPaths):
    A = np.zeros([nPaths])
    Sum = np.zeros([nPaths])
    for ti in Ti:
        idx =(np.abs(time-ti)).argmin()
        temp1 = 1.0/sigma* S[:,idx]*(np.log(S[:,idx]/S0)-(r+0.5*sigma**2.0)*ti)
        Sum = Sum + temp1;
        A = A+ S[:,idx]
    
    A = A /len(Ti)    
    vega  = 1/len(Ti)*np.exp(-r*T)*np.mean((A>K)*Sum)
    return vega

def mainCalculation():
    CP        = OptionType.CALL
    S0        = 100.0
    r         = 0.05
    sigma     = 0.15
    T         = 1.0
    K         = np.array([S0])

    NoOfSteps = 1000
    NoOfPathsMax = 25000;
    
    # Time grid for averaging

    nPeriods = 10
    Ti= np.linspace(0,T,nPeriods)
    
    # Delta estimated by central differences

    dS0 = 1e-04
    np.random.seed(2)
    paths =GeneratePathsGBMEuler(NoOfPathsMax,NoOfSteps,T,r,sigma,S0-dS0)
    S1 = paths["S"]
    time1= paths["time"]
    np.random.seed(2)
    paths =GeneratePathsGBMEuler(NoOfPathsMax,NoOfSteps,T,r,sigma,S0+dS0);
    S2 = paths["S"]
    time2= paths["time"]
    asian1 = AsianOption(S1,time1,Ti,r,T,CP,K,NoOfPathsMax);
    value1 = asian1["value"]
    asian2 = AsianOption(S2,time2,Ti,r,T,CP,K,NoOfPathsMax);
    value2 = asian2["value"]
    delta_Exact = (value2-value1)/(2.0*dS0);
    
    # Vega estimated by central differences

    dsigma = 1e-04
    np.random.seed(2)
    paths =GeneratePathsGBMEuler(NoOfPathsMax,NoOfSteps,T,r,sigma-dsigma,S0)
    S1 = paths["S"]
    time1= paths["time"]
    np.random.seed(2)
    paths =GeneratePathsGBMEuler(NoOfPathsMax,NoOfSteps,T,r,sigma+dsigma,S0);
    S2 = paths["S"]
    time2= paths["time"]
    asian1    =AsianOption(S1,time1,Ti,r,T,CP,K,NoOfPathsMax);
    value1    = asian1["value"]
    asian2    =AsianOption(S2,time2,Ti,r,T,CP,K,NoOfPathsMax);
    value2    = asian2["value"]
    vega_Exact = (value2-value1)/(2.0*dsigma);
    
    # Simulation of the pathwsie sensitivity

    NoOfPathsV = np.round(np.linspace(5,NoOfPathsMax,20))
    deltaPathWiseV = np.zeros(len(NoOfPathsV))
    vegaPathWiseV  = np.zeros(len(NoOfPathsV))
    
    for (idx,nPaths) in enumerate(NoOfPathsV):
        print('Running simulation with {0} paths'.format(nPaths))
        np.random.seed(3)
        paths1 = GeneratePathsGBMEuler(int(nPaths),NoOfSteps,T,r,sigma,S0)
        S    = paths1["S"]
        time = paths1["time"]
        asian = AsianOption(S,time,Ti,r,T,CP,K,int(nPaths))
        A = asian["A"].reshape([int(nPaths),1])
        
        delta_pathwise = PathwiseDelta(S0,A,K,r,T)
        deltaPathWiseV[idx]= delta_pathwise
        
        vega_pathwise = PathwiseVegaAsian(S0,S,time,sigma,K,r,T,Ti,int(nPaths))
        vegaPathWiseV[idx] =vega_pathwise
        
    plt.figure(1)
    plt.grid()
    plt.plot(NoOfPathsV,deltaPathWiseV,'.-r')
    plt.plot(NoOfPathsV,delta_Exact*np.ones([len(NoOfPathsV),1]))
    plt.xlabel('number of paths')
    plt.ylabel('Delta')
    plt.title('Convergence of pathwise delta w.r.t number of paths')
    plt.legend(['pathwise est','exact'])
    
    plt.figure(2)
    plt.grid()
    plt.plot(NoOfPathsV,vegaPathWiseV,'.-r')
    plt.plot(NoOfPathsV,vega_Exact*np.ones([len(NoOfPathsV),1]))
    plt.xlabel('number of paths')
    plt.ylabel('Vega')
    plt.title('Convergence of pathwise vega w.r.t number of paths')
    plt.legend(['pathwise est','exact'])
    
mainCalculation()
