#%%
"""
Created on Feb 06 2019
Call option price and derivatives with respect to strike, dC/dK and d2C/dK2
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 
from mpl_toolkits import mplot3d
from scipy.interpolate import RegularGridInterpolator

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
        
    X[:,0] = np.log(S_0)
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma * sigma) * dt + sigma * (W[:,i+1]-W[:,i])
        time[i+1] = time[i] +dt
        
    # Compute exponent of ABM

    S = np.exp(X)
    paths = {"time":time,"S":S}
    return paths

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

def dCdK(S_0,K,sigma,t,T,r):
    K = np.array(K).reshape([len(K),1])
    C = lambda k: BS_Call_Put_Option_Price(OptionType.CALL,S_0,k,sigma,t,T,r)
    dK = 0.0001 
    return (C(K+dK)-C(K-dK))/(2.0*dK)
    
def d2CdK2(S_0,K,sigma,t,T,r):
    K = np.array(K).reshape([len(K),1])
    dCdK_ = lambda k: dCdK(S_0,k,sigma,t,T,r)
    dK = 0.0001 
    return (dCdK_(K+dK)-dCdK_(K-dK))/(2.0*dK)

def mainCalculation():
    T         = 1.001
    r         = 0.05
    sigma     = 0.4
    s0        = 10.0
         
    # Settings for the plots    

    KGrid     = np.linspace(s0/100.0,1.5*s0,50)
    timeGrid   = np.linspace(0.02,T-0.02,100)
    
    # Prepare the necessary lambda functions

    CallOpt   = lambda t,K : BS_Call_Put_Option_Price(OptionType.CALL,s0,K,sigma,t,T,r)
    dCalldK   = lambda t,K : dCdK(s0,K,sigma,t,T,r)
    d2CalldK2 = lambda t,K : d2CdK2(s0,K,sigma,t,T,r)
    
    # Prepare empty matrices for storing the results

    callOptM   = np.zeros([len(timeGrid),len(KGrid)])
    dCalldKM   = np.zeros([len(timeGrid),len(KGrid)])
    d2CalldK2M = np.zeros([len(timeGrid),len(KGrid)])
    TM         = np.zeros([len(timeGrid),len(KGrid)])
    KM         = np.zeros([len(timeGrid),len(KGrid)])
    
    for i in range(0,len(timeGrid)):
        TM[i,:]         = timeGrid[i]
        KM[i,:]         = KGrid
        callOptM[i,:]   = CallOpt(timeGrid[i],KGrid).transpose()
        dCalldKM[i,:]   = dCalldK(timeGrid[i],KGrid).transpose()
        d2CalldK2M[i,:]   = d2CalldK2(timeGrid[i],KGrid).transpose()
        
    # Plot the call surface.    

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.plot_surface(TM, KM, callOptM, color=[1,0.5,1])
    plt.xlabel('t')
    plt.ylabel('K')
    plt.title('Call option surface')
    
    # Plot the dCdK surface.    

    fig = plt.figure(2)
    ax = fig.gca(projection='3d')
    ax.plot_surface(TM, KM, dCalldKM, color=[1,0.5,1])
    plt.xlabel('t')
    plt.ylabel('K')
    plt.title('dC/dK')
    
    # Plot the d2CdK2 surface.    

    fig = plt.figure(3)
    ax = fig.gca(projection='3d')
    ax.plot_surface(TM, KM, d2CalldK2M, color=[1,0.5,1])
    plt.xlabel('t')
    plt.ylabel('K')
    plt.title('d2C/dK2')
            
mainCalculation()
