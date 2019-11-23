#%%
"""
Created on Jan 23 2019
Paths for the Hull White model, negative paths and 3D Figure
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
from mpl_toolkits import mplot3d

def HWMean(P0T,lambd,eta,T):

    # Time step needed for differentiation

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)

    # Initial interest rate is forward rate at time t->0

    r0 = f0T(0.00001)
    theta = Theta(P0T,eta,lambd)
    zGrid = np.linspace(0.0,T,2500)
    temp =lambda z: theta(z) * np.exp(-lambd*(T-z))
    r_mean = r0*np.exp(-lambd*T) + lambd * integrate.trapz(temp(zGrid),zGrid)
    return r_mean

def HWVar(lambd,eta,T):
    return eta*eta/(2.0*lambd) *( 1.0-np.exp(-2.0*lambd *T))

def HWDensity(P0T,lambd,eta,T):
    r_mean = HWMean(P0T,lambd,eta,T)
    r_var = HWVar(lambd,eta,T)
    return lambda x: st.norm.pdf(x,r_mean,np.sqrt(r_var))
    
def Theta(P0T,eta,lambd):

    # Time step needed for differentiation

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2.0*dt)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) +\
    eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))      
    return theta

def GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambd, eta):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    R = np.zeros([NoOfPaths, NoOfSteps+1])

    # Time step needed for differentiation

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)

    # Initial interest rate is forward rate at time t->0

    r0 = f0T(0.00001)
    R[:,0]=r0
    time = np.zeros([NoOfSteps+1])
    
    theta = Theta(P0T,eta,lambd)    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):

        # making sure that samples from a normal have mean 0 and variance 1

        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        R[:,i+1] = R[:,i] + lambd*(theta(time[i]) - R[:,i]) * dt + eta* (W[:,i+1]-W[:,i])
        time[i+1] = time[i] +dt
        
    # Output

    paths = {"time":time,"R":R}
    return paths

def mainCalculation():
    NoOfPaths = 10
    NoOfSteps = 500
    T         = 50.0
    lambd     = 0.5
    eta       = 0.04
    
    # We define a ZCB curve (obtained from the market)

    P0T = lambda T: np.exp(0.05*T) 

    # Effect of mean reversion lambda

    plt.figure(1) 
    np.random.seed(1)
    Paths = GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambd, eta)
    timeGrid = Paths["time"]
    R = Paths["R"]       
    plt.plot(timeGrid, np.transpose(R))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("R(t)")        
       
    # 3D graph for R(t) for paths with density

    plt.figure(2)
    ax = plt.axes(projection='3d')
    zline = np.zeros([len(timeGrid),1])
    
    # Plot paths

    n = 5
    for i in range(0,n,1):
        y1 = np.squeeze(np.transpose(R[i,:]))
        x1 = timeGrid
        z1 = np.squeeze(zline)
        ax.plot3D(x1, y1, z1, 'blue')
    ax.view_init(50, -170)
    
    Ti = np.linspace(0.5,T,5)
    y1 = np.linspace(np.min(np.min(R)),np.max(np.max(R)),100)
    for ti in Ti:

          # Density for the v(t) process    

        hwDensity = HWDensity(P0T,lambd,eta,ti)
        x1 = np.zeros([len(y1),1]) + ti
        z1 = hwDensity(y1) 
        ax.plot3D(x1, y1, z1, 'red')
    
    # Compute numerical expectation and compare to analytic expression

    EV_num = integrate.trapz(y1*hwDensity(y1) ,y1)
    print("numerical: E[r(t=5)]={0}".format(EV_num))
    print("theoretical: E[r(t=5)]={0}".format( HWMean(P0T,lambd,eta,T)))
mainCalculation()
