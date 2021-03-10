#%%
"""
Created on Nov 10 2019
Fourier approximation of a function: sine expansion
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

i = np.complex(0.0, 1.0) #assigning i=sqrt(-1)

def FourierCosineSeries(y,f,N,a,b):
        
    g = np.zeros(len(y))
    for k in range(0,N):
        A_k = 2.0/(b-a) * integrate.trapz(f(y)*np.cos(k*np.pi*(y-a)/(b-a)),y)
        if k == 0:
            A_k = A_k * 0.5
        g = g + A_k*np.cos(k*np.pi*(y-a)/(b-a)) 
    return g

def FourierSineSeries(y,f,N,a,b):
        
    g = np.zeros(len(y))
    for k in range(0,N):
        B_k = 2.0/(b-a) * integrate.trapz(f(y)*np.sin(k*np.pi*(y-a)/(b-a)),y)
        g = g + B_k*np.sin(k*np.pi*(y-a)/(b-a)) 
    return g

def mainCalculation():
    #define the range for the expansion points
    N = [2,5,10,40,200]
    
    K= 10.0
    a = 0.0
    b = 30.0
    y = np.linspace(a,b,200)
    
    f = lambda x: np.maximum(x-K,0)

    plt.figure(1)    
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('payoff')
    
    
    for i in N:
        f_proxy = FourierSineSeries(y,f,i,a,b)
        plt.plot(y,f_proxy)
    plt.legend(N)
    
mainCalculation()