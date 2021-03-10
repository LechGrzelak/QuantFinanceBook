#%%
"""
Created on 29 Nov 2019
Conditional expectation
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

def GeneratePathsTwoStocksEuler(NoOfPaths,NoOfSteps,T,r,S10,S20,rho,sigma1,sigma2):
    Z1 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    Z2 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W1 = np.zeros([NoOfPaths, NoOfSteps+1])
    W2 = np.zeros([NoOfPaths, NoOfSteps+1])
    # Initialization
    X1 = np.zeros([NoOfPaths, NoOfSteps+1])
    X1[:,0] =np.log(S10)
    X2 = np.zeros([NoOfPaths, NoOfSteps+1])
    X2[:,0] =np.log(S20)
    time = np.zeros([NoOfSteps+1])
    dt = T / float(NoOfSteps)
    
    for i in range(0,NoOfSteps):
        # Making sure that samples from a normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z1[:,i] = (Z1[:,i] - np.mean(Z1[:,i])) / np.std(Z1[:,i])
            Z2[:,i] = (Z2[:,i] - np.mean(Z2[:,i])) / np.std(Z2[:,i])
            Z2[:,i] = rho *Z1[:,i] + np.sqrt(1.0-rho**2.0)*Z2[:,i]
        W1[:,i+1] = W1[:,i] + np.power(dt, 0.5)*Z1[:,i]
        W2[:,i+1] = W2[:,i] + np.power(dt, 0.5)*Z2[:,i]
        X1[:,i+1] = X1[:,i] + (r -0.5*sigma1**2.0)* dt + sigma1 * (W1[:,i+1] -
          W1[:,i])
        X2[:,i+1] = X2[:,i] + (r -0.5*sigma2**2.0)* dt + sigma2 * (W2[:,i+1] -
              W2[:,i])
        time[i+1] = time[i] +dt
        
    # Return stock paths
    paths = {"time":time,"S1":np.exp(X1),"S2":np.exp(X2)}
    return paths

def getEVBinMethod(S,v,NoOfBins):
    if (NoOfBins != 1):
        mat = np.transpose(np.array([S,v]))
        # Sort all the rows according to the first column
        val = mat[mat[:,0].argsort()]
        binSize = int((np.size(S)/NoOfBins))
        expectation = np.zeros([np.size(S),2])
        for i in range(1,binSize-1):
            sample = val[(i-1)*binSize:i*binSize,1]
            expectation[(i-1)*binSize:i*binSize,0] = val[(i-1)*binSize:i*binSize,0]
            expectation[(i-1)*binSize:i*binSize,1] = np.mean(sample)
    return expectation

def main():
    NoOfPaths = 10000
    NoOfSteps = 100
    T         = 5.0
    rho       = -0.5
    y10       = 1.0
    y20       = 0.05
    
    # Set 1
    sigma1    = 0.3
    sigma2    = 0.3
    
    # Set 2
    #sigma1    = 0.9
    #sigma2    = 0.9
    
    paths = GeneratePathsTwoStocksEuler(NoOfPaths, NoOfSteps, T, 0.0, y10, y20, rho, sigma1, sigma2)
    y1 = paths["S1"]
    y2 = paths["S2"]
    
    # Analytical expression for the conditional expectation    
    condE = lambda y1: y20 * (y1/y10)**(rho*sigma2/sigma1)*np.exp(0.5*T*(rho*sigma2*sigma1-sigma2**2*rho**2))
    
    plt.figure(1)
    plt.grid()
    plt.plot(y1[:,-1],y2[:,-1],'.')
    y1Grid = np.linspace(np.min(y1[:,-1]), np.max(y1[:,-1]), 2500)
    plt.plot(y1Grid,condE(y1Grid),'r')

    # Bin method
    E = getEVBinMethod(y1[:,-1], y2[:,-1], 50)
    plt.plot(E[:,0],E[:,1],'k')
    plt.legend(['samples','E[Y2|Y1]-Monte Carlo','E[Y2|Y1]-Analytical'])
    plt.xlabel('Y1')
    plt.ylabel('Y2')
    
    # for y1 = 1.75 we have
    y1 = 1.75
    print("Analytical expression for y1={0} yields E[Y2|Y1={0}] = {1}".format(y1,condE(y1)))
    
    condValueInterp = lambda y1: np.interp(y1,E[:,0],E[:,1])
    print("Monte Carlo, for y1={0} yields E[Y2|Y1={0}] = {1}".format(y1,condValueInterp(y1)))
        
main()