#%%
"""
Created on Thu Dec 12 2018
Shifted GBM and pricing of caplets/floorlets
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 
from mpl_toolkits import mplot3d
import scipy.integrate as integrate
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as optimize

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def GeneratePathsGBMShifted(NoOfPaths,NoOfSteps,T,r,sigma,S_0,shift):
    S_0shift = S_0 + shift
    if (S_0shift < 0.0):
        raise('Shift is too small!')
    
    paths =GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,S_0shift)
    Sshifted = paths["S"] - shift
    time = paths["time"]
    return {"time":time,"S":Sshifted}      

def GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
        
    X[:,0] = np.log(S_0)
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma * sigma) * dt + sigma * (W[:,i+1]-W[:,i])
        time[i+1] = time[i] +dt
        
    # Compute exponent of ABM

    S = np.exp(X)
    paths = {"time":time,"S":S}
    return paths

# Shifted Black-Scholes call option price

def BS_Call_Put_Option_Price_Shifted(CP,S_0,K,sigma,tau,r,shift):
    K_new = K + shift
    S_0_new = S_0 + shift
    return BS_Call_Put_Option_Price(CP,S_0_new,K_new,sigma,tau,r)

# Black-Scholes call option price

def BS_Call_Put_Option_Price(CP,S_0,K,sigma,tau,r):
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * tau) / (sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

# Implied volatility method

def ImpliedVolatilityBlack76(CP,marketPrice,K,T,S_0):

    # To determine initial volatility we define a grid for sigma
    # and interpolate on the inverse function

    sigmaGrid = np.linspace(0.0,2.0,5000)
    optPriceGrid = BS_Call_Put_Option_Price(CP,S_0,K,sigmaGrid,T,0.0)
    sigmaInitial = np.interp(marketPrice,optPriceGrid,sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))
    
    # Use already determined input for the local-search (final tuning)

    func = lambda sigma: np.power(BS_Call_Put_Option_Price(CP,S_0,K,sigma,T,0.0) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-15)
    print("Final volatility = {0}".format(impliedVol))
    return impliedVol

# Implied volatility method

def ImpliedVolatilityBlack76Shifted(CP,marketPrice,K,T,S_0,shift):

    # To determine initial volatility we define a grid for sigma
    # and interpolate on the inverse function

    sigmaGrid = np.linspace(0.0,2.0,5000)
    optPriceGrid = BS_Call_Put_Option_Price_Shifted(CP,S_0,K,sigmaGrid,T,0.0,shift)
    sigmaInitial = np.interp(marketPrice,optPriceGrid,sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))
    
    # Use already determined input for the local-search (final tuning)

    func = lambda sigma: np.power(BS_Call_Put_Option_Price_Shifted(CP,S_0,K,sigma,T,0.0,shift) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-15)
    print("Final volatility = {0}".format(impliedVol))
    return impliedVol


def mainCalculation():
    NoOfPaths = 10000
    NoOfSteps = 500
    T         = 3.0
    sigma     = 0.2
    L0        = -0.05
    shift     = 0.1
    K         = [0.95]
    CP        = OptionType.CALL
    
    P0T =lambda T: np.exp(-0.1*T)
    
    np.random.seed(4)
    Paths = GeneratePathsGBMShifted(NoOfPaths,NoOfSteps,T,0.0,sigma,L0,shift)
    time  = Paths["time"]
    L     = Paths["S"]
    
    print(np.mean(L[:,-1]))
    
    # Plot first few paths

    plt.figure(1)
    plt.plot(time,np.transpose(L[0:20,:]))
    plt.grid()
    
    # Shifted lognormal for different shift parameters

    plt.figure(2)
    shiftV = [1.0,2.0,3.0,4.0,5.0]
    legend = []
    for shiftTemp in shiftV:
        x = np.linspace(-shiftTemp,10,1000)
        lognnormPDF = lambda x,t :  st.lognorm.pdf(x+shiftTemp, scale = np.exp(np.log(L0+shiftTemp) + (- 0.5 * sigma * sigma)*t), s= np.sqrt(t) * sigma)
        pdf_x= lognnormPDF(x,T)
        plt.plot(x,pdf_x)
        legend.append('shift={0}'.format(shiftTemp))
    plt.legend(legend)
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.title('shifted lognormal density')
    plt.grid()
    
    # Call/Put option prices, MC vs. Analytic prices

    plt.figure(3)
    K = np.linspace(-shift,np.abs(L0)*3,25)
    optPriceMCV=np.zeros([len(K),1])
    for idx in range(0,len(K)):
        optPriceMCV[idx] =0.0
        if CP == OptionType.CALL:
            optPriceMCV[idx] = P0T(T)*np.mean(np.maximum(L[:,-1]-K[idx],0.0))
        elif CP == OptionType.PUT:
            optPriceMCV[idx] = P0T(T)*np.mean(np.maximum(K[idx]-L[:,-1],0.0))
    
    optPriceExact = P0T(T)*BS_Call_Put_Option_Price_Shifted(CP,L0,K,sigma,T,0.0,shift)
    plt.plot(K,optPriceMCV)       
    plt.plot(K,optPriceExact,'--r')       
    plt.grid()
    plt.xlabel('strike,K')
    plt.ylabel('option price')
    plt.legend(['Monte Carlo','Exact'])
    
    # Shift effect on option prices

    plt.figure(4)
    legend = []
    for shiftTemp in [0.2,0.3,0.4,0.5]:    
        K = np.linspace(-shiftTemp,np.abs(L0)*6.0,25)
        optPriceExact = P0T(T)*BS_Call_Put_Option_Price_Shifted(CP,L0,K,sigma,T,0.0,shiftTemp)
        plt.plot(K,optPriceExact)       
        legend.append('shift={0}'.format(shiftTemp))
    plt.grid()
    plt.xlabel('strike,K')
    plt.ylabel('option price')
    plt.legend(legend)
    
mainCalculation()
