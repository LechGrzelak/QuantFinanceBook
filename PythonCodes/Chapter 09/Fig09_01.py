#%%
"""
Created on Jan 16 2019
Stochastic integration of int_0^1t^2dW(t)
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

# We simulate paths for dX(t) = t^2 dW(t) with X(t_0)=0

def GenerateMonteCarloPaths(NoOfPaths,NoOfSteps,T,g):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])

    dt = T / float(NoOfSteps)
    t = 0 
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        X[:,i+1] = X[:,i] + g(t) * np.power(dt, 0.5)*Z[:,i]
        t = t + dt
    return X

def mainCalculation():
    NoOfPaths = 2000
    NoOfSteps = 100
    T = 1
    g = lambda t:t*t
    X =GenerateMonteCarloPaths(NoOfPaths,NoOfSteps,T,g)
        
    EX_T = np.mean(X[:,-1])
    VarX_T = np.var(X[:,-1])
    print("E(X(T)) = {0:4e}  and Var(X(T))={1:4f}".format(EX_T,VarX_T))
    plt.figure(1)
    plt.plot(np.linspace(0,T,NoOfSteps+1), np.transpose(X))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("X(t)")
    
    plt.figure(2)
    plt.grid()
    plt.hist(X[:,-1],25)
    

mainCalculation()
