#%%
"""
Created on Thu Jan 03 2019
CGMY and convergence obtained with the COS method
@author: Lech A. Grzelak
"""
import numpy as np
import scipy.stats as st
import enum 
import scipy.optimize as optimize
import scipy.special as stFunc

# Set i= imaginary number

i   = np.complex(0.0,1.0)

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
def CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L):


    # cf   - Characteristic function is a function, in the book denoted by \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # r    - Interest rate (constant)
    # tau  - Time to maturity
    # K    - List of strikes
    # N    - Number of expansion terms
    # L    - Size of truncation domain (typ.:L=8 or L=10)

    # Reshape K to become a column vector

    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    # Assigning i=sqrt(-1)

    i = np.complex(0.0,1.0) 
    x0 = np.log(S0 / K)   
    
    # Truncation domain

    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # Summation from k = 0 to k=N-1

    k = np.linspace(0,N-1,N).reshape([N,1])  
    u = k * np.pi / (b - a);  

    # Determine coefficients for put prices  

    H_k = CallPutCoefficients(OptionType.PUT,a,b,k)   
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))     
    
    # We use the put-call parity for call options

    if CP == OptionType.CALL:
        value = value + S0 - K*np.exp(-r*tau)    
        
    return value

# Determine coefficients for put prices 

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
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/(b - a))
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
    
# Black-Scholes call option price

def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * tau) / float(sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

# Implied volatility method

def ImpliedVolatility(CP,marketPrice,K,T,S_0,r):
    func = lambda sigma: np.power(BS_Call_Option_Price(CP,S_0,K,sigma,T,r) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, 0.7, tol=1e-5)
    #impliedVol = optimize.brent(func, brack= (0.05, 2))
    return impliedVol

def ChFCGMY(r,tau,S0,C,G,M,Y,sigma):
    i = np.complex(0.0,1.0)
    varPhi = lambda u: np.exp(tau * C *stFunc.gamma(-Y)*( np.power(M-i*u,Y) - np.power(M,Y) + np.power(G+i*u,Y) - np.power(G,Y)))
    omega =  -1/tau * np.log(varPhi(-i))
    cF = lambda u: varPhi(u) * np.exp(i*u* (r+ omega -0.5*sigma*sigma)*tau - 0.5*sigma*sigma *u *u *tau)  
    return cF 

def mainCalculation():
    CP  = OptionType.CALL   
       
    # Define the range for the expansion terms

    N = [4,8,16,24,32,36,40,48,64,80,96,128]
    L = 8
        
    # Parameter setting for the CGMY-BM model

    S0    = 100.0
    r     = 0.1
    sigma = 0.2
    C     = 1.0
    G     = 5.0
    M     = 5.0
    Y     = 0.5 
    tau   = 1.0
    K = [S0]
    K = np.array(K).reshape([len(K),1])    
    
    # Characteristic function of the CGMY model

    cf = ChFCGMY(r,tau,S0,C,G,M,Y,sigma)
    
    # Exact option price

    Nexact = np.power(2,14)
    optExact = CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, Nexact, L)
    print("Exact solution for Y= {0} is equal to = {1}".format(Y,np.squeeze(optExact[0])))
    
    for n in N:
        opt = CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, n, L)
        error = np.squeeze(np.abs(optExact-opt))
        print("For {0} expanansion terms the error is {1:2.3e}".format(n,error))
                        
mainCalculation()
