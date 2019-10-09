#%%
"""
Normal density recovery using the COS method- Convergence exercise
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

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
    
def mainCalculation():
    i = np.complex(0.0, 1.0) #assigning i=sqrt(-1)
    
    # Setting for the COS method 

    a = -10
    b = 10
    
    # Define the range for the expansion terms

    N = [2**x for x in range(2,7,1)]
    
    # Parameter setting for normal distribution

    mu = 0
    sigma = 1 
        
    # Define characteristic function for the normal distribution

    cF = lambda u : np.exp(i * mu * u - 0.5 * np.power(sigma,2.0) * np.power(u,2.0));
    
    # Define the domain for density

    x = np.linspace(-5.0,5,1000)
    f_XExact = st.norm.pdf(x,mu,sigma)
    
    plt.figure(1)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("$f_X(x)$")
    for n in N:
        f_X = COSDensity(cF,x,n,a,b)
        error = np.max(np.abs(f_X-f_XExact))
        print("For {0} expanansion terms the error is {1}".format(n,error))
        
        plt.plot(x,f_X)
        plt.legend(["N=%.0f"%N[0],"N=%.0f"%N[1],"N=%.0f"%N[2],"N=%.0f"%N[3],"N=%.0f"%N[4]])
    
    
mainCalculation()
