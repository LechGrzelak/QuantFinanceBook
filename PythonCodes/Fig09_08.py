#%%
"""
Created on Feb 11 2019
3D surface of normal samples with marginal distributions
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from mpl_toolkits import mplot3d

def mainCalculation():
    NoOfPaths = 1000
    
    # Parameters

    mu1    = 5.0
    mu2    = 10.0
    sigma1 = 2.0
    sigma2 = 0.5
    rho    = 0.7
    
    # Generation of random samples

    X = np.random.normal(0.0,1.0,[NoOfPaths,1])
    Y = np.random.normal(0.0,1.0,[NoOfPaths,1])
    X = (X - np.mean(X)) / np.std(X)
    Y = (Y - np.mean(Y)) / np.std(Y)
    
    Y = rho *X + np.sqrt(1.0-rho**2) * Y

    # Adjustment for the mean and variance

    X = sigma1* X + mu1
    Y = sigma2* Y + mu2
    
    # 3D plot

    # First we show the scatter plot with the samples

    plt.figure(1)
    ax = plt.axes(projection='3d')
    Z = np.zeros([NoOfPaths,1])
    ax.plot3D(np.squeeze(X), np.squeeze(Y), np.squeeze(Z),'.')
        
    # Now we add the marginal densities

    x1=np.linspace(st.norm.ppf(0.001, loc=mu1, scale=sigma1),st.norm.ppf(1.0-0.001, loc=mu1, scale=sigma1),30)
    y1=np.min(Y) - 1.0 + np.zeros([len(x1),1])
    z1=st.norm.pdf(x1,loc=mu1, scale=sigma1)
    ax.plot3D(np.squeeze(x1), np.squeeze(y1), np.squeeze(z1), 'blue')
    
    x2=np.linspace(st.norm.ppf(0.001, loc=mu2, scale=sigma2),st.norm.ppf(1.0-0.001, loc=mu2, scale=sigma2),30)
    y2=np.min(X) - 1.0 + np.zeros([len(x2),1])
    z2=st.norm.pdf(x2,loc=mu2, scale=sigma2)
    ax.plot3D(np.squeeze(y2), np.squeeze(x2), np.squeeze(z2), 'red')
    
    ax.view_init(50, -120)
mainCalculation()
