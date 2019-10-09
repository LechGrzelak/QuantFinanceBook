#%%
"""
Created on Thu Dec 12 2018
GBM paths and strike figures
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

def GeneratePathsGBMABM(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
        
    X[:,0] = np.log(S_0)
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma * sigma) * dt + sigma * np.power(dt, 0.5)*Z[:,i]
        time[i+1] = time[i] +dt
        
    # Compute exponent of ABM

    S = np.exp(X)
    paths = {"time":time,"S":S}
    return paths

def mainCalculation():
    np.random.seed(1)
    NoOfPaths = 2
    NoOfSteps = 100
    T         = 1
    r         = 0.05
    sigma     = 1
    S_0       = 100
    K         = 150
    Tcall     = [0.4, 0.6, 0.8]
    
    Paths = GeneratePathsGBMABM(NoOfPaths,NoOfSteps,T,r,sigma,S_0)
    timeGrid = Paths["time"]
    S = Paths["S"]
        
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(S))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.plot(T,K,'ok')

    plt.figure(2)
    plt.plot(timeGrid, np.transpose(S))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.plot(T,K,'ok')
    for ti in Tcall:
        plt.plot(ti*np.ones([300,1]),range(0,300),'--k')
       
mainCalculation()
