#%%
"""
Created on Nov 15 2019
Exercise 9.7: computation of variance of X(t)=W(t)+t/TW(T-t)
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

def GenerateBrownianMotion(NoOfPaths,NoOfSteps,T):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])

    dt = T / float(NoOfSteps)
    t = 0.0
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        t = t + dt
    return W

def mainCalculation():
    NoOfPaths = 1000
    NoOfSteps = 1000
    T = 10.0
    dt = T / float(NoOfSteps)
    W = GenerateBrownianMotion(NoOfPaths,NoOfSteps,T)
    time = np.linspace(0,T,NoOfSteps+1)
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    varX = [0.0]
    t = 0.0
    for i in range(0,NoOfSteps):
        X[:,i+1] = W[:,i] -t/T * W[:,NoOfSteps-i] 
        t = t + dt
        varX.append(np.var(X[:,i+1]))

    plt.figure(1)
    plt.plot(time, np.transpose(X))
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('paths')
    
    plt.figure(2)
    plt.plot(time,varX)
    varExact = lambda t: t-2.0*(t/T)*np.minimum(t,T-t) + (t**2.0)/(T**2.0)*(T-t)
    plt.plot(time,varExact(time),'--r')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('variance')
    plt.legend(['Monte Carlo','Exact'])    
mainCalculation()