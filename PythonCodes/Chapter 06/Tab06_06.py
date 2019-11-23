#%%
"""
Created on Thu Dec 14 2018
CGMY density drecovery with the COS method
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as stFunc

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

def CGMYdensity(y,C,G,M,Y):
    f_y = np.zeros([len(y),1])
    for idx, yi in y:
        if yi >0:
            f_y[idx] = C*(np.exp(-M*np.abs(y))/(np.power(np.abs(y),1.0+Y)))
        elif yi<0:
            f_y[idx] = C*(np.exp(-G*np.abs(y))/(np.power(np.abs(y),1.0+Y)))
    return f_y

def mainCalculation():
    i = np.complex(0.0, 1.0) #assigning i=sqrt(-1)
       
    # Parameter setting for the CGMY-BM model

    S0    = 100
    r     = 0.1
    sigma = 0.2
    C     = 1.0
    G     = 5.0
    M     = 5.0
    t     = 1.0
    
    # Define the range for the expansion terms
    # N = [32, 48, 64, 80, 96, 128]
    # Y = 0.5
    
    N = [8,16, 24,32,40,48]
    Y = 1.5

    # Setting for the COS method

    L = 8
    
    varPhi = lambda u: np.exp(t * C *stFunc.gamma(-Y)*( np.power(M-i*u,Y) - \
                    np.power(M,Y) + np.power(G+i*u,Y) - np.power(G,Y)))
    omega =  -1/t * np.log(varPhi(-i))
    cF = lambda u: varPhi(u) * np.exp(i*u* (np.log(S0)+ r+ omega -0.5*sigma*sigma)*t\
                          - 0.5*sigma*sigma *u *u *t)  

    # Cumulants needed for the integration range

    zeta1 = (np.log(S0)+ r + omega -0.5*sigma*sigma)*t + t*C*Y*stFunc.gamma(-Y)*\
    (np.power(G,Y-1)-np.power(M,Y-1))
    zeta2 = sigma * sigma *t+ t*C*stFunc.gamma(2-Y)*(np.power(G,Y-2) + np.power(M,Y-2))
    zeta4 = t*C*stFunc.gamma(4-Y) * (np.power(G,Y-4) + np.power(M,Y-4))
    
    # Define the COS method integration range

    a = zeta1 - L * np.sqrt(zeta2 + np.sqrt(zeta4))
    b = zeta1 + L * np.sqrt(zeta2 + np.sqrt(zeta4))
    
    # Define the domain for density

    x = np.linspace(-2,12,250)
    
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
        plt.plot(x,f_X)
    
    plt.legend(legendList)
    
    print('exact = {0}'.format(S0*np.exp(r*t)))
mainCalculation()
