#%%
"""
Created on Jan 28 2019
Time Change example
@author: Lech A. Grzelak
"""
import numpy as np
import scipy.stats as st
import scipy.integrate as integrate
import enum 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.interpolate import interp1d 

# Set i= imaginary number

i   = np.complex(0.0,1.0)

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
def GenerateBrownianMotion(NoOfPaths,NoOfSteps,T):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
   
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        time[i+1] = time[i] +dt
        
    # Return stock paths

    paths = {"time":time,"W":W}
    return paths

def mainCalculation():
    NoOfPaths = 1000
    NoOfSteps = 1000
    T   = 5.0
    
    nu  = lambda t: 0.3+np.log(t+0.01)/10.0
        
    # Time-dependent volatility

    np.random.seed(1)
    paths1 = GenerateBrownianMotion(NoOfPaths,NoOfSteps,T)    
    W = paths1["W"]
    time =paths1["time"]   
    
    plt.figure(1)
    plt.plot(time, np.transpose(W[0:50,:]),'b')   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("Brownian motion")
    
    # Here we compute X(t) = mu(t) W(t)

    X=np.zeros([NoOfPaths,NoOfSteps+1])
    for idx, t in enumerate(time):
        X[:,idx] = nu(t) * W[:,idx]
    
    plt.figure(2)
    plt.plot(time, np.transpose(X[0:50,:]),'b')   
    plt.xlabel("time")
    plt.ylabel("X(t)")
    
    # Here we compute Y(t) = W(t*mu^2(t))

    Y=np.zeros([NoOfPaths,NoOfSteps+1])
    
    for idx, t in enumerate(time):
        current_time = (np.abs(time-t*nu(t)*nu(t))).argmin()
        Y[:,idx] = W[:,current_time]
    
    plt.figure(2)
    plt.plot(time, np.transpose(Y[0:50,:]),'r')   
    plt.xlabel("time")
    plt.ylabel("Y(t)")
    plt.grid()
    
    plt.figure(3)
    sns.distplot(X[:,-1], hist = False, kde = True, rug = False, color = 'darkblue',\
                 kde_kws={'linewidth': 3}, rug_kws={'color': 'black'}) 
    sns.distplot(Y[:,-1], hist = False, kde = True, rug = False, color = 'red',\
                 kde_kws={'linewidth': 2}, rug_kws={'color': 'black'}) 
    plt.grid()
    plt.legend(['X(t)=nu(t)W(t)','Y(t)=W(mu^2(t)*t)'])
    

mainCalculation()
