#%%
"""
Created on Nov 15 2019
Exercise 9.9, simulation of the CIR process, Euler vs. AES
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def GeneratePathsCIR_AES(NoOfPaths,NoOfSteps,T,kappa,v0,vbar,gamma):
    V      = np.zeros([NoOfPaths, NoOfSteps+1])
    V[:,0] = v0
    time   = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        V[:,i+1]= CIR_Sample(NoOfPaths,kappa,gamma,vbar,0,dt,V[:,i])
        time[i+1] =time[i] + dt
    # Outputs
    paths = {"time":time,"V":V}
    return paths
        
def GeneratePathsCIREuler(NoOfPaths,NoOfSteps,T,kappa,v0,vbar,gamma):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    V = np.zeros([NoOfPaths, NoOfSteps+1])
    V[:,0]=v0
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        V[:,i+1] = V[:,i] + kappa*(vbar - V[:,i]) * dt + gamma* np.sqrt(V[:,i]) * (W[:,i+1]-W[:,i])
        # We apply here the truncation scheme for negative values
        V[:,i+1] = np.maximum(V[:,i+1],0.0)
        time[i+1] = time[i] +dt
        
    # Outputs
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

def CIRVar(kappa,gamma,vbar,s,t,v_s):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    VarV = c*c*(2.0*delta+4.0*kappaBar)
    return VarV

def CIR_Sample(NoOfPaths,kappa,gamma,vbar,s,t,v_s):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    sample = c* np.random.noncentral_chisquare(delta,kappaBar,NoOfPaths)
    return  sample

def mainCalculation():
    NoOfPaths = 10000
    NoOfSteps = 1000
    T     = 5.0
    kappa = 0.5
    v0    = 0.01
    vbar  = 0.1
    gamma = 0.6
    
    Paths = GeneratePathsCIREuler(NoOfPaths,NoOfSteps,T,kappa,v0,vbar,gamma)
    timeGrid = Paths["time"]
    V = Paths["V"]
    """
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(V),'b')   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("V(t)")
    """
    # AES 
    PathsAES = GeneratePathsCIR_AES(NoOfPaths,NoOfSteps,T,kappa,v0,vbar,gamma)
    timeGrid = PathsAES["time"]
    V_AES = PathsAES["V"]
    
    """
    plt.figure(2)
    plt.plot(timeGrid, np.transpose(V_AES),'b')   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("V(t)")
    """
    
    # Feller condition
    feller = 2.0*kappa*vbar - gamma**2.0
    if feller<0:
        print("Feller condition is NOT satisfied and 2.0*kappa*vbar - gamma**2.0 = {0}".format(feller))
    else:
        print("Feller condition is satisfied and 2.0*kappa*vbar - gamma**2.0 = {0}".format(feller))
    
    # Here we compare expectations and variances for Euler and AES disctretization schemes
    EX_Euler  = np.mean(V,axis=0)
    EX_AES    = np.mean(V_AES,axis=0)
    Var_Euler = np.var(V,axis=0)
    Var_AES   = np.var(V_AES,axis=0)
    
    plt.figure(3)
    plt.plot(timeGrid,CIRMean(kappa,gamma,vbar,v0,timeGrid))
    plt.plot(timeGrid,EX_Euler,'-k')
    plt.plot(timeGrid,EX_AES,'--r')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('E(v(t))')
    plt.legend(['Exact','Euler','AES'])
    plt.title('NoOfSteps = {0}'.format(NoOfSteps))    
    
    plt.figure(4)
    plt.plot(timeGrid,CIRVar(kappa,gamma,vbar,0,timeGrid,v0))
    plt.plot(timeGrid,Var_Euler,'-k')
    plt.plot(timeGrid,Var_AES,'--r')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('Var(v(t))')
    plt.legend(['Exact','Euler','AES'])
    plt.title('NoOfSteps = {0}'.format(NoOfSteps))    
mainCalculation()