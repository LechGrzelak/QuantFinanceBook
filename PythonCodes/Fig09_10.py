#%%
"""
Created on Jan 22 2019
Monte Carlo paths for the Jacobi correlation model
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

def GeneratePathsJacobi(NoOfPaths,NoOfSteps,T,rho0,kappa,gamma,mu):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    Rho = np.zeros([NoOfPaths, NoOfSteps+1])
    Rho[:,0]=rho0
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        Rho[:,i+1] = Rho[:,i] + kappa*(mu - Rho[:,i]) * dt + gamma* np.sqrt(1.0 -Rho[:,i]*Rho[:,i]) * (W[:,i+1]-W[:,i])
        
        # Handling of the boundary conditions to ensure paths stay within the [-1,1] range

        Rho[:,i+1] = np.maximum(Rho[:,i+1],-1.0)
        Rho[:,i+1] = np.minimum(Rho[:,i+1],1.0)
        time[i+1] = time[i] +dt
        
    # Output

    paths = {"time":time,"Rho":Rho}
    return paths

def mainCalculation():
    NoOfPaths = 15
    NoOfSteps = 500
    T         = 5
    kappa     = 0.25
    rho0      = 0.0
    kappa     = 0.1
    gamma     = 0.6
    mu        = 0.5
    
    # Check the boundaries:

    if kappa > np.max([gamma*gamma/(1.0-mu),gamma*gamma/(1.0+mu)]):
        print("Boundry is NOT attainable")
    else:
        print("Boundry is attainable")
        
    Paths = GeneratePathsJacobi(NoOfPaths,NoOfSteps,T,rho0,kappa,gamma,mu)   
    timeGrid = Paths["time"]
    Rho = Paths["Rho"]
    
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(Rho),'b')   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("Rho(t)")
           
mainCalculation()
