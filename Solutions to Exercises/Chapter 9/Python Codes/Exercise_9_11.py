#%%
"""
Created on Nov 09 2019
Exercise 9.11: Monte Carlo for the VG, CGMY and NIG models: comparison with the COS method
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as stFunc
import enum

# set i= imaginary number
i   = np.complex(0.0,1.0)

# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L):
    # cf   - characteristic function as a functon, in the book denoted as \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # r    - interest rate (constant)
    # tau  - time to maturity
    # K    - list of strikes
    # N    - Number of expansion terms
    # L    - size of truncation domain (typ.:L=8 or L=10)  
        
    # reshape K to a column vector
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    #assigning i=sqrt(-1)
    i = np.complex(0.0,1.0) 
    x0 = np.log(S0 / K)   
    
    # truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # sumation from k = 0 to k=N-1
    k = np.linspace(0,N-1,N).reshape([N,1])  
    u = k * np.pi / (b - a);  

    # Determine coefficients for Put Prices  
    H_k = CallPutCoefficients(CP,a,b,k)   
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))     
    return value

# Determine coefficients for Put Prices 
def CallPutCoefficients(CP,a,b,k):
    if CP==OptionType.CALL:                  
        c = 0.0
        d = b
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k),1])
        else:
            H_k      = 2.0 / (b - a) * (Chi_k - Psi_k)  
    elif CP==OptionType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k      = 2.0 / (b - a) * (- Chi_k + Psi_k)               
    
    return H_k    

def Chi_Psi(a,b,c,d,k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * \
                (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0)) 
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)  - np.cos(k * np.pi 
                  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * 
                        (d - a) / (b - a))   - k * np.pi / (b - a) * np.sin(k 
                        * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    
    value = {"chi":chi,"psi":psi }
    return value

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

def CfNIG(t,delta,alpha,beta,sigma,S0,r):
    varPhi = lambda u: np.exp(t * delta * (np.sqrt(alpha**2.0-beta**2.0)-np.sqrt(alpha**2.0-(beta**2.0+i*u)*(beta**2.0+i*u))))
    cF = lambda u: varPhi(u) * np.exp( i*u* r *t - 0.5*sigma*sigma *u *u *t)  
    return cF

def CfCGMY(t,C,G,M,Y,sigma,S0,r):
    i = np.complex(0.0, 1.0) #assigning i=sqrt(-1)
    varPhi = lambda u: np.exp(t * C *stFunc.gamma(-Y)*( np.power(M-i*u,Y) - np.power(M,Y) + np.power(G+i*u,Y) - np.power(G,Y)))
    omega =  -1/t * np.log(varPhi(-i))
    cF = lambda u: varPhi(u) * np.exp(i*u* ( r+ omega -0.5*sigma*sigma)*t - 0.5*sigma*sigma *u *u *t)  
    return cF

def GeneratePathsVG(NoOfPaths,T,r,S0,sigmaVG,beta,theta):
    G     = np.random.gamma(T/beta,beta,NoOfPaths)
    Z     = np.random.normal(0.0,1.0,NoOfPaths)
    X_bar = theta * G + sigmaVG * np.sqrt(G) * Z
    omega = 1.0 / beta * np.log(1.0-beta*theta-0.5*beta*sigmaVG**2.0) 
    mu_VG =  r+ omega
    X     = mu_VG * T  + X_bar    
    S     = S0 * np.exp(X)    
    return S    

def CfVG(t,r,beta,theta,sigma,S0):
    i = np.complex(0.0,1.0)
    omega = 1/beta * np.log(1.0-beta*theta-0.5*beta*sigma*sigma)
    mu = r + omega
    cf = lambda u: np.exp(i*u*mu*t)*np.power(1.0-i*beta*theta*u+0.5*beta*sigma*sigma*u*u,-t/beta)
    return cf

def EUOptionPriceFromMCPathsGeneralized(CP,S,K,T,r):
    # S is a vector of Monte Carlo samples at T
    result = np.zeros([len(K),1])
    if CP == OptionType.CALL:
        for (idx,k) in enumerate(K):
            result[idx] = np.exp(-r*T)*np.mean(np.maximum(S-k,0.0))
    elif CP == OptionType.PUT:
        for (idx,k) in enumerate(K):
            result[idx] = np.exp(-r*T)*np.mean(np.maximum(k-S,0.0))
    return result


def mainCalculation():
    #define the range for the expansion points
    N = 4096
    
    CP = OptionType.CALL
    
    # setting for the VG-BM model
    S0    = 40.0
    r     = 0.1
    sigma = 0.1
    
    beta  = 0.2
    theta = -0.14
    T     = 4.0
    
    # CGMY
    C = 1.0
    G = 1.0
    M = 5.0
    Y = 0.5
    
    # NIG
    delta = .1
    
    alpha = 5.5
    beta = 0.01

    # setting for the COS method
    L = 10
    
    cF_VG = CfVG(T,r,beta,theta,sigma,S0)
    cF_CGMY = CfCGMY(T,C,G,M,Y,sigma,S0,r)
    cF_NIG = CfNIG(T,delta,alpha,beta,sigma,S0,r)
   
    # option pricing part
    K= np.linspace(0.1,S0*2,25)
    K = np.array(K).reshape([len(K),1])
    
    call_VG = CallPutOptionPriceCOSMthd(cF_VG,CP,S0,r,T,K,N,L)
    #call_CGMY = CallPutOptionPriceCOSMthd(cF_CGMY,OptionType.CALL,S0,r,T,K,N,L)
    #call_NIG = CallPutOptionPriceCOSMthd(cF_NIG,OptionType.CALL,S0,r,T,K,N,L)
    
    plt.figure(1)
    plt.plot(K,call_VG)
    #plt.plot(K,call_CGMY)
    #plt.plot(K,call_NIG)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('call option value')
    #plt.legend(['VG','CGMY','NIG'])
    
    NoOfPaths = 1000000
    S_VG = GeneratePathsVG(NoOfPaths,T,r,S0,sigma,beta,theta)
    optEUR_VG = EUOptionPriceFromMCPathsGeneralized(CP,S_VG,K,T,r)
    plt.plot(K,optEUR_VG,'--r')   
    
mainCalculation()