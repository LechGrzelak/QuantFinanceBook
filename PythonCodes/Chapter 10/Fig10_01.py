#%%
"""
Created on Jan 2 2019
The Heston model and pricing of forward start options
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 
import scipy.optimize as optimize

# Set i= imaginary number

i   = np.complex(0.0,1.0)

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def CallPutOptionPriceCOSMthd_FrwdStart(cf,CP,r,T1,T2,K,N,L):

    # cf   - Characteristic function as a functon, in the book denoted by \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # r    - Interest rate (constant)
    # tau  - Time to maturity
    # K    - List of strikes
    # N    - Number of expansion terms
    # L    - Size of truncation domain (typ.:L=8 or L=10)

    tau = T2 - T1

    # Reshape K to become a column vector

    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    # Adjust strike price

    K = K + 1.0
    
    # Assigning i=sqrt(-1)

    i = np.complex(0.0,1.0) 
    x0 = np.log(1.0 / K)   
    
    # Truncation domain

    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # Summation from k = 0 to k=N-1

    k = np.linspace(0,N-1,N).reshape([N,1])  
    u = k * np.pi / (b - a);  

    # Determine coefficients for put prices  

    H_k = CallPutCoefficients(CP,a,b,k)   
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    value = np.exp(-r * T2) * K * np.real(mat.dot(temp))     
    return value

# Determine coefficients for put and call prices 

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
    
# Forward start Black-Scholes option price

def BS_Call_Option_Price_FrwdStart(K,sigma,T1,T2,r):
    if K is list:
        K = np.array(K).reshape([len(K),1])
    K = K + 1.0 
    tau = T2 - T1
    d1    = (np.log(1.0 / K) + (r + 0.5 * np.power(sigma,2.0))* tau) / (sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    value = np.exp(-r*T1) * st.norm.cdf(d1) - st.norm.cdf(d2) * K * np.exp(-r * T2)
    return value

# Implied volatility for the forward start call option

def ImpliedVolatility_FrwdStart(marketPrice,K,T1,T2,r):

    # To determine initial volatility we define a grid for sigma
    # and interpolate on the inverse function

    sigmaGrid = np.linspace(0,2,200)
    optPriceGrid = BS_Call_Option_Price_FrwdStart(K,sigmaGrid,T1,T2,r)
    sigmaInitial = np.interp(marketPrice,optPriceGrid,sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))
    
    # Use already determined input for the local-search (final tuning)

    func = lambda sigma: np.power(BS_Call_Option_Price_FrwdStart(K,sigma,T1,T2,r) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-15)
    print("Final volatility = {0}".format(impliedVol))
    return impliedVol

def ChFHestonModelForwardStart(r,T1,T2,kappa,gamma,vbar,v0,rho):
    i = np.complex(0.0,1.0)
    tau = T2 - T1
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*np.exp(-D1(u)*tau)))\
        *(kappa-gamma*rho*i*u-D1(u))

    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method

    A  = lambda u: r*i*u*tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*rho*i*u-D1(u))\
        - 2*kappa*vbar/gamma/gamma*np.log((1.0-g(u)*np.exp(-D1(u)*tau))/(1.0-g(u)))
    c_bar = lambda t1,t2: gamma*gamma/(4.0*kappa) * (1.0 - np.exp(-kappa*(t2-t1)))
    delta = 4.0*kappa*vbar/gamma/gamma
    kappa_bar = lambda t1, t2: 4.0*kappa*v0*np.exp(-kappa*(t2-t1))/(gamma*gamma*(1.0-np.exp(-kappa*(t2-t1))))
    term1 = lambda u: A(u) + C(u)*c_bar(0.0,T1)*kappa_bar(0.0,T1)/(1.0-2.0*C(u)*c_bar(0.0,T1))
    term2 = lambda u: np.power(1.0/(1.0-2.0*C(u)*c_bar(0.0,T1)),0.5*delta)
    cf = lambda u: np.exp(term1(u)) * term2(u)
    return cf

def mainCalculation():
    CP  = OptionType.CALL
    r   = 0.00
    
    TMat1=[[1.0,3.0],[2.0,4.0],[3.0, 5.0],[4.0, 6.0]]
    TMat2=[[1.0,2.0],[1.0,3.0],[1.0, 4.0],[1.0, 5.0]]
        
    K = np.linspace(-0.4,4.0,50)
    K = np.array(K).reshape([len(K),1])
    
    N = 500
    L = 10
        
    # Heston model parameters

    kappa = 0.6
    gamma = 0.2
    vbar  = 0.1
    rho   = -0.5
    v0    = 0.05   

    plt.figure(1)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    legend = []
    for T_pair in TMat1:
       T1= T_pair[0]
       T2= T_pair[1]
       cf = ChFHestonModelForwardStart(r,T1,T2,kappa,gamma,vbar,v0,rho)

       # Forward-start option by the COS method

       valCOS = CallPutOptionPriceCOSMthd_FrwdStart(cf,CP,r,T1,T2,K,N,L)

       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           IV[idx] = ImpliedVolatility_FrwdStart(valCOS[idx],K[idx],T1,T2,r)
       plt.plot(K,IV*100.0)
       legend.append('T1={0} & T2={1}'.format(T1,T2))
    plt.legend(legend)
    
    plt.figure(2)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    legend = []
    for T_pair in TMat2:
       T1= T_pair[0]
       T2= T_pair[1]
       cf = ChFHestonModelForwardStart(r,T1,T2,kappa,gamma,vbar,v0,rho)
       
       # Forward-start option by the COS method

       valCOS = CallPutOptionPriceCOSMthd_FrwdStart(cf,CP,r,T1,T2,K,N,L)
    
       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           IV[idx] = ImpliedVolatility_FrwdStart(valCOS[idx],K[idx],T1,T2,r)
       plt.plot(K,IV*100.0)
       legend.append('T1={0} & T2={1}'.format(T1,T2))
       
    plt.legend(legend)

mainCalculation()
