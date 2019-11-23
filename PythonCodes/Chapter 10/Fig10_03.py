#%%
"""
Created on Feb 13 2019
Stopping time simulation for Brownian motion
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

def GeneratePathsBM(NoOfPaths,NoOfSteps,T):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths,NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.sqrt(dt)*Z[:,i]
        time[i+1] = time[i] +dt
        
    paths = {"time":time,"W":W}
    return paths

def mainCalculation():
    NoOfPaths = 3
    NoOfSteps = 500
    T = 10.0

    # Define the barrier

    B =2.0 
    np.random.seed(2)
    Paths = GeneratePathsBM(NoOfPaths,NoOfSteps,T)
    timeGrid = Paths["time"]
    W = Paths["W"]
    
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(W))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("W(t)")
   
    hittingTime=[]
    for i in range(0,NoOfPaths):
        idx = np.where(W[i,:]>B)
        if idx[0]!=np.array([]):
            plt.plot(timeGrid[idx[0][0]],W[i,idx[0][0]],'or')
            hittingTime.append(timeGrid[idx[0][0]])
            
    plt.figure(2)
    plt.hist(hittingTime,50)
    plt.grid()
mainCalculation()
