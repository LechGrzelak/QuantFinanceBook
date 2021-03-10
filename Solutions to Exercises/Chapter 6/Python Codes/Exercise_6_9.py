#%%
"""
Created on Jan 29 2020
Call-Put parity and effect on convergence
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
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

def CfVG(t,r,beta,theta,sigma,S0):
    i = np.complex(0.0,1.0)
    omega = 1/beta * np.log(1.0-beta*theta-0.5*beta*sigma*sigma)
    mu = r + omega
    cf = lambda u: np.exp(i*u*mu*t)*np.power(1.0-i*beta*theta*u+0.5*beta*sigma*sigma*u*u,-t/beta)
    return cf

def mainCalculation():
    #define the range for the expansion points
    N = 700
    
    # setting for the VG-BM model
    S0    = 1.0
    r     = 0.05
    sigma = 0.1
    
    beta  = 0.2
    theta = -0.14
    T     = 2.0       
    beta  = 0.1

    # setting for the COS method
    L = 15
    
    cF_VG = CfVG(T,r,beta,theta,sigma,S0)
   
    # option pricing part
    K= np.linspace(0.5,S0*3.0,25)
    K = np.array(K).reshape([len(K),1])
    
    call_VG = CallPutOptionPriceCOSMthd(cF_VG,OptionType.CALL,S0,r,T,K,N,L)
    
    put_VG   = CallPutOptionPriceCOSMthd(cF_VG,OptionType.PUT,S0,r,T,K,N,L)
    call_VG2 = put_VG + S0 - K*np.exp(-r*T)    
        
    plt.figure(1)
    plt.plot(K,call_VG)
    plt.plot(K,call_VG2,'--r')
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('call option value')
    plt.legend(['Call Option, VG','Call Option, VG with call-put parity'])
    plt.title('Call-Put parity and convergence')
    
    # Case for the Black-Scholes model
    # Definition of the characteristic function for GBM, this is an input
    # for the COS method
    # Note that the Chf does not include "+iuX(t_0)" as this coefficient
    # is already included in the evaluation
    # In the book we denote this function by \varphi(u)
    cF_BS = lambda u: np.exp((r - 0.5 * np.power(sigma,2.0)) * i * u * T - 0.5
                          * np.power(sigma, 2.0) * np.power(u, 2.0) * T)
    
    # option pricing part
    K= np.linspace(0.5,S0*3.0,25)
    K = np.array(K).reshape([len(K),1])
   
    call_BS = CallPutOptionPriceCOSMthd(cF_BS,OptionType.CALL,S0,r,T,K,N,L)
    
    put_BS   = CallPutOptionPriceCOSMthd(cF_BS,OptionType.PUT,S0,r,T,K,N,L)
    call_BS2 = put_BS + S0 - K*np.exp(-r*T)    
    
    plt.figure(2)
    plt.plot(K,call_BS)
    plt.plot(K,call_BS2,'--r')
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('call option value')
    plt.legend(['Call Option, BS','Call Option, BS with call-put parity'])
    plt.title('Call-Put parity and convergence')
    
mainCalculation()