#%%
"""
Created on Thu Nov 28 2018
Normal density and effect of the model parameter variations
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def PlotNormalDensity():    
    
    muV = [-1.0, -0.5, 0.0, 0.5, 1.0]
    sigmaV = [0.25, 0.75, 1.25, 1.75]
    
    # Effect of mu for given sigma

    plt.figure(1)
    plt.grid()
    plt.xlabel('x')
    plt.xlabel('PDF')
    x = np.linspace(-5.0, 5.0, 250)
    
    for mu in muV:
        plt.plot(x,st.norm.pdf(x,mu,1.0))
    plt.legend(['mu =-1.0','mu =-0.5','mu =0.0','mu =0.5','mu =1.0'])
    
    # Effect of sigma for given mu

    plt.figure(2)
    plt.grid()
    plt.xlabel('x')
    plt.xlabel('PDF')   
    for sigma in sigmaV:
        plt.plot(x,st.norm.pdf(x,0,sigma))
    plt.legend(['sigma =0.25','sigma =0.75','sigma =1.25','sigma =1.75',])
PlotNormalDensity()
