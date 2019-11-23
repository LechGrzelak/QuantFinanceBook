#%%
"""
Created on Jan 20 2019
Error distribution for different discretization schemes
@author: Lech A. Grzelak
"""
import numpy as np
import scipy.stats as st
import enum 
import matplotlib.pyplot as plt

# Set i= imaginary number

i   = np.complex(0.0,1.0)

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
def GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
   
    # Approximation

    S1 = np.zeros([NoOfPaths, NoOfSteps+1])
    S1[:,0] =S_0
    
    # Exact

    S2 = np.zeros([NoOfPaths, NoOfSteps+1])
    S2[:,0] =S_0
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        S1[:,i+1] = S1[:,i] + r * S1[:,i]* dt + sigma * S1[:,i] * (W[:,i+1] - W[:,i])
        S2[:,i+1] = S2[:,i] * np.exp((r - 0.5*sigma*sigma) *dt + sigma * (W[:,i+1] - W[:,i]))
        time[i+1] = time[i] +dt
        
    # Return S1 and S2

    paths = {"time":time,"S1":S1,"S2":S2}
    return paths

def GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
   
    # Approximation

    S1 = np.zeros([NoOfPaths, NoOfSteps+1])
    S1[:,0] =S_0
    
    # Exact

    S2 = np.zeros([NoOfPaths, NoOfSteps+1])
    S2[:,0] =S_0
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        S1[:,i+1] = S1[:,i] + r * S1[:,i]* dt + sigma * S1[:,i] * (W[:,i+1] - W[:,i]) \
                    + 0.5 * sigma * sigma * S1[:,i] * (np.power(W[:,i+1] - W[:,i],2.0) - dt)
        S2[:,i+1] = S2[:,i] * np.exp((r - 0.5*sigma*sigma) *dt + sigma * (W[:,i+1] - W[:,i]))
        time[i+1] = time[i] +dt
        
    # Return S1 and S2

    paths = {"time":time,"S1":S1,"S2":S2}
    return paths

def mainCalculation():
    NoOfSteps = 25
    T = 1
    r = 0.06
    sigma = 0.3
    S_0 = 50
    NoOfSteps =10
    
    # Simulated paths

    NoOfPaths = 10000
    PathsEuler = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S_0)    
    PathsMilstein = GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S_0)    
    
    errorEuler = PathsEuler["S1"][:,-1] - PathsEuler["S2"][:,-1]
    errorMilstein = PathsMilstein["S1"][:,-1] - PathsMilstein["S2"][:,-1]
    
    # Histogram, Euler error

    plt.figure(1)
    plt.hist(errorEuler,50)
    plt.grid()

    # Histogram, Milstein error

    plt.figure(2)
    plt.hist(errorMilstein,50)
    plt.grid()
    
mainCalculation()
