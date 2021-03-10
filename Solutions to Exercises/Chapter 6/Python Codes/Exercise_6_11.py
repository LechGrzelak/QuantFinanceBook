#%%
"""
Created on Nov 10 2019
COS method with filter
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as stFunc

i = np.complex(0.0, 1.0) #assigning i=sqrt(-1)

def COSDensity(cf,x,N,a,b):
    i = np.complex(0.0,1.0) #assigning i=sqrt(-1)
    k = np.linspace(0,N-1,N)
    u = np.zeros([1,N])
    u = k * np.pi / (b-a)
        
    #F_k coefficients
    F_k    = 2.0 / (b - a) * np.real(cf(u) * np.exp(-i * u * a));
    F_k[0] = F_k[0] * 0.5; # adjustment for the first term
    
    #Final calculation
    f_X = np.matmul(F_k , np.cos(np.outer(u, x - a )))
        
    # we output only the first row
    return f_X

def COSDensityFilter(cf,x,N,a,b,p):
    i = np.complex(0.0,1.0) #assigning i=sqrt(-1)
    k = np.linspace(0,N-1,N)
    u = np.zeros([1,N])
    u = k * np.pi / (b-a)
        
    #F_k coefficients
    F_k    = 2.0 / (b - a) * np.real(cf(u) * np.exp(-i * u * a));
    F_k[0] = F_k[0] * 0.5; # adjustment for the first term
    
    # filter
    epsilon =1e-16
    alpha=-np.log(epsilon)
    s = lambda x: np.exp(-alpha*x**p)
    filt = s(k/N)
    F_k = F_k *filt
    
    #Final calculation
    f_X = np.matmul(F_k , np.cos(np.outer(u, x - a )))
        
    # we output only the first row
    return f_X

def pdfVG(y, t,beta,theta,sigma):
    term1 = lambda z: 1.0/ (sigma*np.sqrt(2.0*np.pi*z))*np.exp(-(y-theta*z)**2.0/(2.0*sigma**2.0*z))
    term2 = lambda z: (z**(t/beta-1)*np.exp(-z/beta))/(beta**(t/beta)*stFunc.gamma(t/beta))
    z = np.linspace(0.000001,10,100)
    return integrate.trapz(term1(z)*term2(z),z)

def CfVG(t,beta,theta,sigma):
    i = np.complex(0.0,1.0)
    cf = lambda u: np.power(1.0-i*beta*theta*u+0.5*beta*sigma*sigma*u*u,-t/beta)
    return cf


def mainCalculation():
    #define the range for the expansion points
    N = 500
    
    # setting for the VG-BM model
    sigma = 0.12
    
    beta  = 0.2
    theta = -0.14
    T     = 1
 
    # setting for the COS method
    L = 10
    
    cF_VG = CfVG(T,beta,theta,sigma)
          
    # define the COS method integration range
    a = - L * np.sqrt(T)
    b = L * np.sqrt(T)
    
    # define domain for density
    x = np.linspace(-1,0.5,501)
            
    plt.figure(1)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("$f_X(x)$")
    
    f_X_VG = COSDensity(cF_VG,x,N,a,b)
    p = 6.0
    f_X_VG_filter = COSDensityFilter(cF_VG,x,N,a,b,p)
        
    plt.figure(1)
    plt.plot(x,f_X_VG)
    plt.plot(x,f_X_VG_filter)
    plt.legend(['VG','VG_filter'])
    plt.title('density recovery (log space)')
    print(integrate.trapz(f_X_VG,x))
    print(integrate.trapz(f_X_VG_filter,x))
    
    # exact 
    f_y=[]
    for xi in x:
        f_y.append(pdfVG(xi, T,beta,theta,sigma))
    
    plt.figure(2)
    plt.plot(x,np.log(np.abs(f_y-f_X_VG)))
    plt.plot(x,np.log(np.abs(f_y-f_X_VG_filter)))
    plt.legend(['VG','VG_filter'])
    plt.title('Error')
    plt.grid()
    
mainCalculation()