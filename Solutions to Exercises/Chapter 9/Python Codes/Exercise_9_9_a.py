#%%
"""
Created on Nov 15 2019
P[v_{i+1}<0] where v(t) follows the CIR process
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def mainCalculation():
    NoOfSteps = 5
    T     = 1.0
    kappa = 0.4
    vbar  = 0.1
    gamma = 0.25
    dt = T/NoOfSteps
        
    # Feller condition
    feller = 2.0*kappa*vbar - gamma**2.0
    if feller<0:
        print("Feller condition is NOT satisfied and 2.0*kappa*vbar - gamma**2.0 = {0}".format(feller))
    else:
        print("Feller condition is satisfied and 2.0*kappa*vbar - gamma**2.0 = {0}".format(feller))
    
    leg = []
    for NoOfSteps in range(2,6,1):
        dt = T/NoOfSteps
        F = lambda v_i: st.norm.cdf( (-v_i-kappa*(vbar-v_i)*dt)/(gamma*np.sqrt(v_i)*dt))       
        v_i = np.linspace(0.01,0.2,100)
        leg.append('NoOfSteps = {0}'.format(NoOfSteps))
        plt.plot(v_i,F(v_i))
    
    plt.legend(leg)
    plt.grid()
    plt.xlabel('v_i')
    plt.ylabel('P(v_{i+1}<0)')
mainCalculation()