#%%
"""
Created on Thu Jan 16 2019
VG density drecovery with the COS method
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def COSDensity(cf,x,N,a,b):
    i = np.complex(0.0,1.0) #assigning i=sqrt(-1)
    k = np.linspace(0,N-1,N)
    u = np.zeros([1,N])
    u = k * np.pi / (b-a)
        
    # F_k coefficients

    F_k    = 2.0 / (b - a) * np.real(cf(u) * np.exp(-i * u * a));
    F_k[0] = F_k[0] * 0.5; # adjustment for the first term
    
    # Final calculation

    f_X = np.matmul(F_k , np.cos(np.outer(u, x - a )))
        
    # We output only the first row

    return f_X

def VGdensity(y,C,G,M,Y):
    f_y = np.zeros([len(y),1])
    for idx, yi in y:
        if yi >0:
            f_y[idx] = C*(np.exp(-M*np.abs(y))/(np.power(np.abs(y),1.0+Y)))
        elif yi<0:
            f_y[idx] = C*(np.exp(-G*np.abs(y))/(np.power(np.abs(y),1.0+Y)))
    return f_y

def CfVG(t,r,beta,theta,sigma,S0):
    i = np.complex(0.0,1.0)
    omega = 1/beta * np.log(1.0-beta*theta-0.5*beta*sigma*sigma)
    mu = r + omega
    cf = lambda u: np.exp(i*u*mu*t+i*u*np.log(S0))*np.power(1.0-i*beta*theta*u+0.5*beta*sigma*sigma*u*u,-t/beta)
    return cf

def mainCalculation():

    # Define the range for the expansion terms

    N = [ 16, 32, 64, 4096]
    
    # Parameter setting for the VG model

    S0    = 100
    r     = 0.1
    sigma = 0.12
    beta  = 0.2
    theta = -0.14
    T     = 1.0

    # Setting for the COS method

    L = 150    
    cF = CfVG(T,r,beta,theta,sigma,S0)
    
    # Cumulants needed for the integration range

    omega = 1/beta * np.log(1.0-beta*theta-0.5*beta*sigma*sigma)
    zeta1 = (np.log(S0) - omega +theta)*T;
    zeta2 = (sigma*sigma+beta*theta*theta)*T;
    zeta4 = 3*(np.power(sigma,4.0)*beta+2*np.power(theta,4.0)*np.power(beta,3.0)+4*sigma*sigma*theta*theta*beta*beta)*T;
    
    # Define the COS method integration range

    a = zeta1 - L * np.sqrt(zeta2 + np.sqrt(zeta4))
    b = zeta1 + L * np.sqrt(zeta2 + np.sqrt(zeta4))
    
    # Define domain for density

    x = np.linspace(.5,10,5000)
    y = np.exp(x)

    # Define a reference; the solution for a large number of terms is our reference

    f_XExact = COSDensity(cF,x, np.power(2, 14), a,b)
            
    plt.figure(1)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("$f_X(x)$")
    legendList = []
    
    for n in N:
        f_X = COSDensity(cF,x,n,a,b)
        error = np.max(np.abs(f_X-f_XExact))
        print("For {0} expanansion terms the error is {1}".format(n,error))
        legendList.append(["N={0}".format(n)])
        print("Integral over the density ={0}".format(integrate.trapz(f_X,x)))
        plt.plot(x,f_X)
        f_Y = 1/y * f_X
        print("E[Y] ={0}".format(integrate.trapz(y*f_Y,y)))
    
    plt.legend(legendList)
    
    print('exact = {0}'.format(S0*np.exp(r*T)))
mainCalculation()
