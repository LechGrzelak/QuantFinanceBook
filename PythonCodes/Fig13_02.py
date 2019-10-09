#%%
"""
Created on Mar 01 2019
The SZHW model and pricing of a diversification product
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import enum 

# Set i= imaginary number

i   = np.complex(0.0,1.0)

# Time step needed for differentiation

dt = 0.0001
 
# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def HW_theta(lambd,eta,P0T):
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))
    return theta

def HW_A(lambd,eta,P0T,T1,T2):
    tau = T2-T1
    zGrid = np.linspace(0.0,tau,250)
    B_r = lambda tau: 1.0/lambd * (np.exp(-lambd *tau)-1.0)
    theta = HW_theta(lambd,eta,P0T)    
    temp1 = lambd * integrate.trapz(theta(T2-zGrid)*B_r(zGrid),zGrid)
    
    temp2 = eta*eta/(4.0*np.power(lambd,3.0)) * (np.exp(-2.0*lambd*tau)*(4*np.exp(lambd*tau)-1.0) -3.0) + eta*eta*tau/(2.0*lambd*lambd)
    
    return temp1 + temp2

def HW_B(lambd,eta,T1,T2):
    return 1.0/lambd *(np.exp(-lambd*(T2-T1))-1.0)

def HW_ZCB(lambd,eta,P0T,T1,T2,rT1):
    B_r = HW_B(lambd,eta,T1,T2)
    A_r = HW_A(lambd,eta,P0T,T1,T2)
    return np.exp(A_r + B_r *rT1)

def EUOptionPriceFromMCPathsGeneralizedStochIR(CP,S,K,T,M):

    # S is a vector of Monte Carlo samples at T

    result = np.zeros([len(K),1])
    if CP == OptionType.CALL:
        for (idx,k) in enumerate(K):
            result[idx] = np.mean(1.0/M*np.maximum(S-k,0.0))
    elif CP == OptionType.PUT:
        for (idx,k) in enumerate(K):
            result[idx] = np.mean(1.0/M*np.maximum(k-S,0.0))
    return result

def GeneratePathsSZHWEuler(NoOfPaths,NoOfSteps,P0T,T,S0,sigma0,sigmabar,kappa,gamma,lambd,eta,Rxsigma,Rxr,Rsigmar):    

    # Time step needed for differentiation

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    
    # Initial interest rate is forward rate at time t->0

    r0 = f0T(0.00001)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))      
    
    # Empty containers for the Brownian motion

    Wx = np.zeros([NoOfPaths, NoOfSteps+1])
    Wsigma = np.zeros([NoOfPaths, NoOfSteps+1])
    Wr = np.zeros([NoOfPaths, NoOfSteps+1])
    
    Sigma = np.zeros([NoOfPaths, NoOfSteps+1])
    X     = np.zeros([NoOfPaths, NoOfSteps+1])
    R     = np.zeros([NoOfPaths, NoOfSteps+1])
    M_t   = np.ones([NoOfPaths,NoOfSteps+1])
    R[:,0]     = r0 
    Sigma[:,0] = sigma0
    X[:,0]     = np.log(S0)
    
    dt = T / float(NoOfSteps)    
    cov = np.array([[1.0, Rxsigma,Rxr],[Rxsigma,1.0,Rsigmar], [Rxr,Rsigmar,1.0]])
      
    time = np.zeros([NoOfSteps+1])    

    for i in range(0,NoOfSteps):
        Z = np.random.multivariate_normal([.0,.0,.0],cov,NoOfPaths)
        if NoOfPaths > 1:
            Z[:,0] = (Z[:,0] - np.mean(Z[:,0])) / np.std(Z[:,0])
            Z[:,1] = (Z[:,1] - np.mean(Z[:,1])) / np.std(Z[:,1])
            Z[:,2] = (Z[:,2] - np.mean(Z[:,2])) / np.std(Z[:,2])
            
        Wx[:,i+1]     = Wx[:,i]     + np.power(dt, 0.5)*Z[:,0]
        Wsigma[:,i+1] = Wsigma[:,i] + np.power(dt, 0.5)*Z[:,1]
        Wr[:,i+1]     = Wr[:,i]     + np.power(dt, 0.5)*Z[:,2]

        # Euler discretization

        R[:,i+1]     = R[:,i] + lambd*(theta(time[i]) - R[:,i])*dt + eta * (Wr[:,i+1]-Wr[:,i])
        M_t[:,i+1]   = M_t[:,i] * np.exp(0.5*(R[:,i+1] + R[:,i])*dt)
        Sigma[:,i+1] = Sigma[:,i] + kappa*(sigmabar - Sigma[:,i])*dt + gamma* (Wsigma[:,i+1]-Wsigma[:,i])                        
        X[:,i+1]     = X[:,i] + (R[:,i] - 0.5*Sigma[:,i]**2.0)*dt + Sigma[:,i] * (Wx[:,i+1]-Wx[:,i])                        
        time[i+1] = time[i] +dt
        
        # Moment matching component, i.e. ensure that E(S(T)/M(T))= S0

        a = S0 / np.mean(np.exp(X[:,i+1])/M_t[:,i+1])
        X[:,i+1] = X[:,i+1] + np.log(a)
        
    paths = {"time":time,"S":np.exp(X),"M_t":M_t,"R":R}
    return paths

def DiversifcationPayoff(P0T,S_T,S0,r_T,M_T,T,T1,lambd,eta,omegaV):
    P_T_T1= HW_ZCB(lambd,eta,P0T,T,T1,r_T)
    P_0_T1= P0T(T1)
    
    value =np.zeros(omegaV.size)
    for (idx,omega) in enumerate(omegaV):
        payoff = omega * S_T/S0 + (1.0-omega) * P_T_T1/P_0_T1
        value[idx] = np.mean(1/M_T*np.maximum(payoff,0.0))
    return value

def mainCalculation():

    # HW model parameter settings

    lambd = 1.12
    eta   = 0.02
    S0    = 100.0
    
    # Fixed mean reversion parameter

    kappa   = 0.5

    # Diversification product

    T  = 9.0
    T1 = 10.0
    
    # We define a ZCB curve (obtained from the market)

    P0T = lambda T: np.exp(-0.033*T) 
     
    # Range of the weighting factor

    omegaV=  np.linspace(-3.0,3.0,50)

    # Monte Carlo setting

    NoOfPaths =5000
    NoOfSteps = int(100*T)
    
    # The SZHW model parameters
    # The parameters can be obtained by the calibration of the SZHW model and 
    # varying the correlation rhoxr.
  
    parameters=[{"Rxr":0.0,"sigmabar":0.167,"gamma":0.2,"Rxsigma":-0.850,"Rrsigma":-0.008,"kappa":0.5,"sigma0":0.035},
                {"Rxr":-0.7,"sigmabar":0.137,"gamma":0.236,"Rxsigma":-0.381,"Rrsigma":-0.339,"kappa":0.5,"sigma0":0.084},
                {"Rxr":0.7,"sigmabar":0.102,"gamma":0.211,"Rxsigma":-0.850,"Rrsigma":-0.340,"kappa":0.5,"sigma0":0.01}]
    
    legend = []
    for (idx,par) in enumerate(parameters):
        sigma0  = par["sigma0"]
        gamma   = par["gamma"]
        Rrsigma = par["Rrsigma"]
        Rxsigma = par["Rxsigma"]
        Rxr     = par["Rxr"]
        sigmabar= par["sigmabar"]
    
        # Generate MC paths

        np.random.seed(1)
        paths = GeneratePathsSZHWEuler(NoOfPaths,NoOfSteps,P0T,T,S0,sigma0,sigmabar,kappa,gamma,lambd,eta,Rxsigma,Rxr,Rrsigma)
        S = paths["S"]
        M = paths["M_t"]
        R = paths["R"]
    
        S_T= S[:,-1]
        R_T= R[:,-1]
        M_T= M[:,-1]
        value_0 = DiversifcationPayoff(P0T,S_T,S0,R_T,M_T,T,T1,lambd,eta,omegaV)
    
        # Reference with rho=0.0

        if Rxr==0.0:
            refR0 = value_0
            
        plt.figure(1)
        plt.plot(omegaV,value_0)
        legend.append('par={0}'.format(idx))
        
        plt.figure(2)
        plt.plot(omegaV,value_0/refR0)
    
    plt.figure(1)
    plt.grid()      
    plt.legend(legend)
    plt.figure(2)
    plt.grid()      
    plt.legend(legend)
    
mainCalculation()
