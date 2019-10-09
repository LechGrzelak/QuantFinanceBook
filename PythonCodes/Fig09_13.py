#%%
"""
Created on Thu Nov 27 2018
Brownian Bridge simulation for a given two points 
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

def GenerateBrownianBridge(NoOfPaths,NoOfSteps,T,a,b,sigma):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    B = np.zeros([NoOfPaths, NoOfSteps+1])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    t = np.zeros([NoOfSteps+1])
    B[:,0]=a        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] +  np.power(dt, 0.5)*Z[:,i]
        B[:,i+1] = B[:,i] +   (b-B[:,i])/(T -t[i])*dt + sigma*(W[:,i+1]-W[:,i])
        t[i+1] = t[i] +dt
        
    paths = {"time":t,"B":B}
    return paths

def mainCalculation():
    NoOfPaths = 6
    NoOfSteps = 2000
    T = 2.0
    V_t0 = 0.1
    V_tN = 0.2
    sigma = 0.03
    
    Paths = GenerateBrownianBridge(NoOfPaths,NoOfSteps,T,V_t0,V_tN,sigma)
    timeGrid = Paths["time"]
    B = Paths["B"]

    plt.figure(1)
    plt.plot(timeGrid, np.transpose(B),'-b')   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("B(t)")
    
                    
mainCalculation()
