#%%
"""
Created on Jan 23 2019
Simulation from the CIR process V(T)|V(t_0) with the QE scheme
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm

def FirstApproach(m,s2):
    b2 = 2.0*m*m/s2 - 1.0 + np.sqrt(2.0*m*m/s2)*np.sqrt(2.0*m*m/s2-1.0)
    b  = np.sqrt(b2)
    a  = m /(1.0+b2)
    return a,b

def SecondApproach(m,s2):
    c = (s2/m/m-1.0)/(s2/m/m+1.0)
    d = (1.0-c)/m
    return c,d

def CIRCDF(kappa,gamma,vbar,s,t,v_s):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    cdf =lambda x: st.ncx2.cdf(x/c,delta,kappaBar)
    return cdf

def CIRDensity(kappa,gamma,vbar,s,t,v_s):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    ncx2PDF = lambda x : 1.0/c * st.ncx2.pdf(x/c,delta,kappaBar)
    return ncx2PDF

def CIRMean(kappa,gamma,vbar,s,t,v_s):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    return c*(delta + kappaBar)

def CIRVar(kappa,gamma,vbar,s,t,v_s):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    VarV = c*c*(2.0*delta+4.0*kappaBar)
    return VarV

def mainCalculation():

    # Number of Monte Carlo samples

    NoOfSamples = 100000
    
    # Maturity times

    T = 3
    
    # QE scheme: we take 1.5 as the boundary, switching  point

    aStar = 1.5

    # CIR parameter settings

    gamma  = 0.2 # 0.6
    vbar   = 0.05
    v0     = 0.3
    kappa  = 0.5
    
    # Mean and the variance for the CIR process

    m  = CIRMean(kappa,gamma,vbar,0.0,T,v0)
    s2 = CIRVar(kappa,gamma,vbar,0.0,T,v0)
    
    # QE simulation of the samples for the V(T)|V(t_0)

    if (s2/m/m < aStar):

        # a and b - First approach

        a,b = FirstApproach(m,s2)
        Z = np.random.normal(0.0,1.0,[NoOfSamples,1])
        Z = (Z-np.mean(Z))/np.std(Z)
        V = a * np.power(b+Z,2.0)
    else:

        # c & d - Second approach

        c,d = SecondApproach(m,s2)
        U = np.random.uniform(0.0,1.0,[NoOfSamples,1])
        V = 1.0/d * np.log((1.0-c)/(1.0-U))
        V[U<c] = 0.0

    # Histogram of V

    plt.figure(1)
    plt.hist(V,50)
    plt.grid()
   
    # CDF for V

    plt.figure(2)
    ecdf = sm.distributions.ECDF(np.squeeze(V))
    x = np.linspace(-0.001, max(V),500)
    y = ecdf(x)
    plt.step(x, y)
    cdf = CIRCDF(kappa,gamma,vbar,0.0,T,v0)
    plt.plot(x,cdf(x),'--r')
    plt.grid()
    plt.legend(['emirical CDF','theo CDF'])
    plt.xlabel('v')
    plt.ylabel('CDF')
    
    
mainCalculation()
