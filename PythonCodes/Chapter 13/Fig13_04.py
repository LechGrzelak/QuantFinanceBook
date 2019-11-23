#%%
"""
Created on Feb 01 2019
Comparison of the volatilities for the Heston and the Schöbel-Zhu model
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
from mpl_toolkits import mplot3d
import seaborn as sns

def GeneratePathsOrnsteinUhlenbecEuler(NoOfPaths,NoOfSteps,T,kappa,sigma0,sigmaBar,gamma):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    sigma = np.zeros([NoOfPaths, NoOfSteps+1])
    sigma[:,0]=sigma0
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        sigma[:,i+1] = sigma[:,i] + kappa*(sigmaBar - sigma[:,i]) * dt + gamma \
                    * (W[:,i+1]-W[:,i])
        time[i+1] = time[i] +dt
        
    # Output

    paths = {"time":time,"sigma":sigma}
    return paths

def GeneratePathsCIREuler(NoOfPaths,NoOfSteps,T,kappa,v0,vbar,gamma):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    V = np.zeros([NoOfPaths, NoOfSteps+1])
    V[:,0]=v0
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        V[:,i+1] = V[:,i] + kappa*(vbar - V[:,i]) * dt + gamma* np.sqrt(V[:,i])\
                    * (W[:,i+1]-W[:,i])

        # We apply here the truncation scheme for negative values

        V[:,i+1] = np.maximum(V[:,i+1],0.0)
        time[i+1] = time[i] +dt
        
   # Output

    paths = {"time":time,"V":V}
    return paths

def CIRDensity(kappa,gamma,vbar,s,t,v_s):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    ncx2PDF = lambda x : 1.0/c * st.ncx2.pdf(x/c,delta,kappaBar)
    return ncx2PDF

def CIRMean(kappa,gamma,vbar,v0,T):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= lambda s,t: 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = lambda s,t,v_s: 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*\
                                                   (1.0-np.exp(-kappa*(t-s))))
    return c(0,T)*(delta + kappaBar(0.0,T,v0))

def mainCalculation():
    NoOfPaths = 25000
    NoOfSteps = 500
    T     = 2.0
    
    # Parameters, Heston

    v0    =0.063
    vbar  =0.063
    
    # Set 1

    kappa = 1.2
    gamma = 0.1
    
    # Set 2

    #kappa = 0.25
    #gamma = 0.63   
    
    # Volatility for the Heston model, sqrt(v(T))

    PathsVolHeston = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,T,kappa,v0,vbar,gamma)
    timeGridHeston = PathsVolHeston["time"]
    V = PathsVolHeston["V"]
    volHeston = np.sqrt(V)
    
    # Volatility for the Schöbel-Zhu model, 
    # We perform moment matching SZ-H, with the assumption kappa =1.0 and sigma0= sqrt(V0)

    sigma0 = np.sqrt(v0)
    kappa = 1.0
    sigmaBar = (np.mean(volHeston[:,-1])-sigma0*np.exp(-T))/(1.0 - np.exp(-T))
    gamma = np.sqrt(2.0*np.var(volHeston[:,-1])/(1.0-np.exp(-2.0*T)))
    PathsVolSZ = GeneratePathsOrnsteinUhlenbecEuler(NoOfPaths,NoOfSteps,T,kappa,sigma0,sigmaBar,gamma)
    volSZ = PathsVolSZ["sigma"]
        
    plt.figure(1)
    plt.plot(timeGridHeston, np.transpose(volHeston[0:50,:]),'b')   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("V(t)")
    
    plt.figure(2)
    sns.distplot(volHeston[:,-1],bins=25)
    sns.distplot(volSZ[:,-1],bins=25)
    plt.grid()
    plt.legend(['sqrt(V(T))','sigma(T)'])
    
mainCalculation()
