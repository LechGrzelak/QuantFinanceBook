#%%
"""
Created on Apr 03 2019
The Heston model and the nonparametric method for estimation of conditional expectation E[V|S=si]
The method is compared to regression method and the 2D COS method
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import enum 

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def ChFHestonModel(r,tau,kappa,gamma,vbar,v0,rho):
    i = np.complex(0.0,1.0)
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*np.exp(-D1(u)*tau)))\
        *(kappa-gamma*rho*i*u-D1(u))

    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method

    A  = lambda u: r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*rho*i*u-D1(u))\
        - 2*kappa*vbar/gamma/gamma*np.log((1.0-g(u)*np.exp(-D1(u)*tau))/(1.0-g(u)))

    # Characteristic function for the Heston model    

    cf = lambda u: np.exp(A(u) + C(u)*v0)
    return cf 

def CIR_Sample(NoOfPaths,kappa,gamma,vbar,s,t,v_s):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    sample = c* np.random.noncentral_chisquare(delta,kappaBar,NoOfPaths)
    return  sample

def GeneratePathsHestonAES(NoOfPaths,NoOfSteps,T,r,S_0,kappa,gamma,rho,vbar,v0):    
    Z1 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W1 = np.zeros([NoOfPaths, NoOfSteps+1])
    V = np.zeros([NoOfPaths, NoOfSteps+1])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    V[:,0]=v0
    X[:,0]=np.log(S_0)
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z1[:,i] = (Z1[:,i] - np.mean(Z1[:,i])) / np.std(Z1[:,i])
        W1[:,i+1] = W1[:,i] + np.power(dt, 0.5)*Z1[:,i]
        
        # Exact samples for the variance process

        V[:,i+1] = CIR_Sample(NoOfPaths,kappa,gamma,vbar,0,dt,V[:,i])
        k0 = (r -rho/gamma*kappa*vbar)*dt
        k1 = (rho*kappa/gamma -0.5)*dt - rho/gamma
        k2 = rho / gamma
        X[:,i+1] = X[:,i] + k0 + k1*V[:,i] + k2 *V[:,i+1] + np.sqrt((1.0-rho**2)*V[:,i])*(W1[:,i+1]-W1[:,i])
        time[i+1] = time[i] +dt
        
    # Compute exponent

    S = np.exp(X)
    paths = {"time":time,"S":S,"V":V}
    return paths

def getEVBinMethod(S,v,NoOfBins):
    if (NoOfBins != 1):
        mat  = np.transpose(np.array([S,v]))
       
        # Sort all the rows according to the first column

        val = mat[mat[:,0].argsort()]
        
        binSize = int((np.size(S)/NoOfBins))
         
        expectation = np.zeros([np.size(S),2])

        for i in range(1,binSize-1):
            sample = val[(i-1)*binSize:i*binSize,1]
            expectation[(i-1)*binSize:i*binSize,0] =val[(i-1)*binSize:i*binSize,0]
            expectation[(i-1)*binSize:i*binSize,1] =np.mean(sample)
        return expectation

def mainCalculation():
    NoOfPaths = 10000
    NoOfSteps = 50
    NoOfBins    = 25
    
    # Heston model parameters

    gamma = 0.1
    kappa = 0.5
    vbar  = 0.07
    rho   = -0.7
    v0    = 0.07
    T     = 1.0
    S_0   = 1.0
    r     = 0.0
        
    # Reference solution with the COS method

    cf = ChFHestonModel(r,T,kappa,gamma,vbar,v0,rho)
    
    # Almost exact simulation

    pathsAES = GeneratePathsHestonAES(NoOfPaths,NoOfSteps,T,r,S_0,kappa,gamma,rho,vbar,v0)
    S = pathsAES["S"][:,-1]
    V = pathsAES["V"][:,-1]
    
    # The scatter plot

    plt.figure(1)
    plt.grid()
    plt.xlabel('S')
    plt.ylabel('V')
    plt.plot(S,V,'.r')
    
    # Non-parametric part

    f = getEVBinMethod(S,V,NoOfBins)
    plt.plot(f[:,0],f[:,1],'k')
            
mainCalculation()
