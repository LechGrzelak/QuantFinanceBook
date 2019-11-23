#%%
"""
Created on Jan 16 2019
Convergence of E(W(t_i+1)-W(t_i))^2 and Var(W(t_i+1)-W(t_i))^2 with respect to
the number of the discretization intervals
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

def mainCalculation():
    NoOfPaths = 50000
    
    mV = []
    vV = []
    T=1.0
    for m in range(2,60,1):
        t1 = 1.0*T/m;
        t2 = 2.0*T/m;
        W_t1 = np.sqrt(t1)* np.random.normal(0.0,1.0,[NoOfPaths,1])
        W_t2 = W_t1 + np.sqrt(t2-t1)* np.random.normal(0.0,1.0,[NoOfPaths,1])
        X = np.power(W_t2-W_t1,2.0);
        mV.append(np.mean(X))
        vV.append(np.var(X))
    
    plt.figure(1)
    plt.plot(range(2,60,1), mV)
    plt.plot(range(2,60,1), vV,'--r')
    plt.grid()
    plt.legend(['E(W(t_i+1)-W(t_i))^2','Var(W(t_i+1)-W(t_i))^2'])
    
mainCalculation()