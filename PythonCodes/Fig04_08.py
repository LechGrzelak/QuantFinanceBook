#%%
"""
Created on Thu Nov 15 2018
Implied volatilities for different sets of model parameters in the SABR model 
(Hagan's et al. formula)
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters for Hagan's SABR approximation formula

CP    = 'c'
beta  = 0.5
alpha = 0.3
rho   = -0.5
gamma = 0.4
f_0   = 1.0
T     = 1.0

def HaganImpliedVolatility(K,T,f,alpha,beta,rho,gamma):

    # We make sure that the input is of array type

    if type(K) == float:
        K = np.array([K])
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    z        = gamma/alpha*np.power(f*K,(1.0-beta)/2.0)*np.log(f/K);
    x_z      = np.log((np.sqrt(1.0-2.0*rho*z+z*z)+z-rho)/(1.0-rho))
    A        = alpha/(np.power(f*K,((1.0-beta)/2.0))*(1.0+np.power(1.0-beta,2.0)/24.0*
                               np.power(np.log(f/K),2.0)+np.power((1.0-beta),4.0)/1920.0*
                               np.power(np.log(f/K),4.0)))
    B1       = 1.0 + (np.power((1.0-beta),2.0)/24.0*alpha*alpha/(np.power((f*K),
                1-beta))+1/4*(rho*beta*gamma*alpha)/(np.power((f*K),
                             ((1.0-beta)/2.0)))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T
    impVol   = A*(z/x_z) * B1
    B2 = 1.0 + (np.power(1.0-beta,2.0)/24.0*alpha*alpha/
                (np.power(f,2.0-2.0*beta))+1.0/4.0*(rho*beta*gamma*
                alpha)/np.power(f,(1.0-beta))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T;

    # Special treatment for ATM strike price

    impVol[np.where(K==f)] = alpha / np.power(f,(1-beta)) * B2;
    return impVol

K    = np.linspace(0.1,3,100)

# Figure 1, effect of parameter beta on the implied volatility

Beta = [0.1, 0.3, 0.5, 0.75, 1]
legendL  = []
plt.figure(1)
for betaTemp in Beta:
    iv = HaganImpliedVolatility(K,T,f_0,alpha,betaTemp,rho,gamma)*100.0
    plt.plot(K,iv)
    legendL.append(('Beta= {0:.1f}').format(betaTemp))
  
plt.xlabel('K')
plt.ylabel('Implied Volatilities [%]')
plt.grid()
plt.legend(legendL)

# Figure 2, effect of alpha on the implied volatility

Alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
legendL  = []
plt.figure(2)
for alphaTemp in Alpha:
    iv = HaganImpliedVolatility(K,T,f_0,alphaTemp,beta,rho,gamma)*100.0
    plt.plot(K,iv)
    legendL.append(('Alpha= {0:.1f}').format(alphaTemp))
  
plt.xlabel('K')
plt.ylabel('Implied Volatilities [%]')
plt.grid()
plt.legend(legendL)

# Figure 3, effect of rho on the implied volatility

Rho = [-0.9, -0.45, 0.0, 0.45, 0.9]
legendL  = []
plt.figure(3)
for rhoTemp in Rho:
    iv = HaganImpliedVolatility(K,T,f_0,alpha,beta,rhoTemp,gamma)*100.0
    plt.plot(K,iv)
    legendL.append(('Rho= {0:.1f}').format(rhoTemp))
  
plt.xlabel('K')
plt.ylabel('Implied Volatilities [%]')
plt.grid()
plt.legend(legendL)

# Figure 4, effect of gamma on the implied volatility

Gamma = [0.1, 0.3, 0.5, 0.7, 0.9]
legendL  = []
plt.figure(4)
for gammaTemp in Gamma:
    iv = HaganImpliedVolatility(K,T,f_0,alpha,beta,rho,gammaTemp)*100.0
    plt.plot(K,iv)
    legendL.append(('Gamma= {0:.1f}').format(gammaTemp))
  
plt.xlabel('K')
plt.ylabel('Implied Volatilities [%]')
plt.grid()
plt.legend(legendL)
