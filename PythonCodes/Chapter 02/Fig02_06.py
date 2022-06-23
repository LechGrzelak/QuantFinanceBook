#%%
"""
Created on Thu Nov 29 2018
Maximum Likelihood Estimation for the Tesla stock
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt


# Market Data - PLEASE COPY these from the corresponding MATLAB CODE

dateS = []
S     = []

def MLE(dateS,S):
    X     = np.log(S)
    plt.plot(dateS,S)
    plt.grid()
    plt.xlabel('days')
    plt.ylabel('Stock Value')

    # Maximum Likelihood Estimation of mu

    dt = 1
    m  = len(dateS)
    mu = 1/(m * dt) * (X[-1]-X[0])
    
    # Maximum Likelihood Estimation of sigma

    s = 0;
    for i in range(0,len(X)-1):
        s = s + np.power(X[i+1]-X[i]-mu*dt,2)
    sigma = np.sqrt(s/(m*dt))
    print('Estimated parameters are: mu = {0} and sigma = {1}'.format(mu,sigma))
    
    # Monte Carlo simulation -- Forecasting

    NoOfPaths    = 10; # plot 10 paths
    NoOfSteps    = 160; # for about 0.5 year
    Z            =  np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    Xsim         =  np.zeros([NoOfPaths,NoOfSteps])
    Xsim[:,0]    =  np.log(S[-1])

    for i in range(0,NoOfSteps-1):
        Z[:,i]         = (Z[:,i]-np.mean(Z[:,i]))/np.std(Z[:,i])
        Xsim[:,i+1]    = Xsim[:,i] + mu*dt + sigma*np.sqrt(dt)*Z[:,i]
    plt.plot(range(dateS[-1],dateS[-1]+NoOfSteps),np.exp(np.transpose(Xsim)),'-r')
MLE(dateS,S)
