#%%
"""
Created on Feb 09 2019
Expectation of the square root of the CIR process
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
from mpl_toolkits import mplot3d
import seaborn as sns

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
        V[:,i+1] = V[:,i] + kappa*(vbar - V[:,i]) * dt + gamma* np.sqrt(V[:,i]) * (W[:,i+1]-W[:,i])
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
    kappaBar = lambda s,t,v_s: 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    return c(0,T)*(delta + kappaBar(0.0,T,v0))

def meanSqrtV_1(kappa,v0,vbar,gamma):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= lambda t: 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t)))
    kappaBar = lambda t: 4.0*kappa*v0*np.exp(-kappa*t)/(gamma*gamma*(1.0-np.exp(-kappa*t)))
    result= lambda t: np.sqrt(c(t) *(kappaBar(t)-1.0 +delta + delta/(2.0*(delta + kappaBar(t)))))
    return result

def meanSqrtV_2(kappa,v0,vbar,gamma):
    a = np.sqrt(vbar-gamma**2.0/(8.0*kappa))
    b = np.sqrt(v0)-a
    temp = meanSqrtV_1(kappa,v0,vbar,gamma)
    epsilon1 = temp(1)
    c = -np.log(1.0/b  *(epsilon1-a))
    return lambda t: a + b *np.exp(-c*t)

def mainCalculation():
    NoOfPaths = 25000
    NoOfSteps = 500
    T         = 5.0       
    
    Parameters1 ={"kappa":1.2,"gamma":0.1,"v0":0.03,"vbar":0.04}
    Parameters2 ={"kappa":1.2,"gamma":0.1,"v0":0.035,"vbar":0.02}
    Parameters3 ={"kappa":1.2,"gamma":0.2,"v0":0.05,"vbar":0.02}
    Parameters4 ={"kappa":0.8,"gamma":0.25,"v0":0.15,"vbar":0.1}
    Parameters5 ={"kappa":1.0,"gamma":0.2,"v0":0.11,"vbar":0.06}
    
    ParV = [Parameters1, Parameters2, Parameters3, Parameters4, Parameters5]
    for par in ParV:
        kappa = par["kappa"]
        gamma = par["gamma"]
        v0    = par["v0"]
        vbar  = par["vbar"]
        
        # Volatility for the Heston model, sqrt(v(T))

        PathsVolHeston = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,T,kappa,v0,vbar,gamma)
        time   = PathsVolHeston["time"]
        time2  = np.linspace(0.0,T,20)
        V      = PathsVolHeston["V"]
        Vsqrt  = np.sqrt(V)
        EsqrtV = Vsqrt.mean(axis=0)
        
        plt.figure(1)
        plt.plot(time,EsqrtV,'b')
        approx1 = meanSqrtV_1(kappa,v0,vbar,gamma)
        approx2 = meanSqrtV_2(kappa,v0,vbar,gamma)
        plt.plot(time,approx1(time),'--r')
        plt.plot(time2,approx2(time2),'.k')
    
    plt.xlabel('time')
    plt.ylabel('E[sqrt(V(t))]')
    plt.grid()
        
mainCalculation()
