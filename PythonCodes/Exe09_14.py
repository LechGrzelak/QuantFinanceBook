#%%
"""
Created on 09 Mar 2019
Pathwise estimation for dV / drho for a modified asset-or-nothing digital option 
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
        
        X1[:,i+1] = X1[:,i] + (r -0.5*sigma1**2.0)* dt + sigma1 * (W1[:,i+1] - W1[:,i])
        X2[:,i+1] = X2[:,i] + (r -0.5*sigma2**2.0)* dt + sigma2 * (W2[:,i+1] - W2[:,i])
        time[i+1] = time[i] +dt
        
    # Return stock paths

    paths = {"time":time,"S1":np.exp(X1),"S2":np.exp(X2)}
    return paths

def PathwiseRho(S10,S20,sigma1,sigma2,rho,S1,S2,K,r,T):
    W1 = 1.0/sigma1*(np.log(S1[:,-1]/S10)-(r-0.5*sigma1**2.0)*T)
    W2 = 1.0/(sigma2*np.sqrt(1.0-rho**2.0))*(np.log(S2[:,-1]/S20)-(r-0.5*sigma2**2.0)*T- sigma2*rho*W1)
    dVdrho = np.exp(-r*T)*np.mean((S1[:,-1]>K)*S2[:,-1]*(sigma2*W1-sigma2*rho/(np.sqrt(1.0-rho**2.0))*W2))
    return dVdrho

def AssetOfNothingPayoff(S1,S2,K,T,r):
    optValue = np.zeros([len(K),1])
    for (idx,k) in enumerate(K):
        optValue[idx] = np.exp(-r*T)*np.mean(S2[:,-1]*(S1[:,-1]>k))
    return optValue

def mainCalculation():
   S10       = 1.0
   S20       = 1.0
   r         = 0.06
   sigma1    = 0.3
   sigma2    = 0.2
   T         = 1.0
   K         = [S10]
   rho       = 0.7

   NoOfSteps = 1000
   
   #% Estimator of the exact rho computed with finite differences

   drho = 1e-5
   np.random.seed(1)
   paths1 = GeneratePathsTwoStocksEuler(20000,NoOfSteps,T,r,S10,S20,rho-drho,sigma1,sigma2)     
   S1 = paths1["S1"]
   S2 = paths1["S2"]
   optValue1 = AssetOfNothingPayoff(S1,S2,K,T,r)
   np.random.seed(1)
   paths2 = GeneratePathsTwoStocksEuler(20000,NoOfSteps,T,r,S10,S20,rho+drho,sigma1,sigma2)     
   S1 = paths2["S1"]
   S2 = paths2["S2"]
   optValue2 = AssetOfNothingPayoff(S1,S2,K,T,r)
   exact_rho = (optValue2- optValue1)/(2.0*drho) 
   print('Reference dV/dRho = {0}'.format(exact_rho[0][0]))

    # Preparation for the simulation

   NoOfPathsV = np.round(np.linspace(5,20000,20))
   rhoPathWiseV = np.zeros([len(NoOfPathsV),1])

   for (idx,nPaths) in enumerate(NoOfPathsV):
       print('Running simulation with {0} paths'.format(nPaths))
              
       np.random.seed(6)
       paths = GeneratePathsTwoStocksEuler(int(nPaths),NoOfSteps,T,r,S10,S20,rho,sigma1,sigma2)         
       S1= paths["S1"]
       S2= paths["S2"]
       
       # dVdrho -- pathwise

       rho_pathwise = PathwiseRho(S10,S20,sigma1,sigma2,rho,S1,S2,K,r,T)
       rhoPathWiseV[idx]= rho_pathwise;

   plt.figure(1)
   plt.grid()
   plt.plot(NoOfPathsV,rhoPathWiseV,'.-r')
   plt.plot(NoOfPathsV,exact_rho*np.ones([len(NoOfPathsV),1]))
   plt.xlabel('number of paths')
   plt.ylabel('dV/drho')
   plt.title('Convergence of pathwise sensitivity to rho w.r.t number of paths')
   plt.legend(['pathwise est','exact'])
    
 
mainCalculation()
