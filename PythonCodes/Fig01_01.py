#%%
"""
Created on Thu Nov 28 2018
Characteristic function and density for normal(10,1)
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from mpl_toolkits import mplot3d

def plotNormalPDF_CDF_CHF():
    mu    = 10.0
    sigma = 1.0
    i     = np.complex(0,1)
    chf   = lambda u: np.exp(i * mu * u - sigma * sigma * u * u / 2.0)
    pdf   = lambda x: st.norm.pdf(x,mu,sigma)
    cdf   = lambda x: st.norm.cdf(x,mu,sigma)

    x = np.linspace(5,15,100)
    u = np.linspace(0,5,250)
    
    # Figure 1, PDF

    plt.figure(1)
    plt.plot(x,pdf(x))
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('PDF')
                     
    # Figure 2, CDF

    plt.figure(2)
    plt.plot(x,cdf(x))
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('CDF')     
    
    # Figure 3, CHF

    plt.figure(3)
    ax = plt.axes(projection='3d')
    chfV = chf(u)
    
    x = np.real(chfV)
    y = np.imag(chfV)
    ax.plot3D(u, x, y, 'blue')
    ax.view_init(30, -120)
    
plotNormalPDF_CDF_CHF()
