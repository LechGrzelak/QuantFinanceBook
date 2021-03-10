#%%
"""
Created on Thu Nov 30 2018
Merton Model and implied volatilities obtained with the COS method compared to 
analytical solution.
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 
import scipy.optimize as optimize

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
    
# Black-Scholes Call option price
def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):
    
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
    func = lambda sigma: np.power(BS_Call_Option_Price(CP,S_0,K,sigma,T,r)\
                                  - marketPrice,1.0)
    impliedVol = optimize.newton(func, 0.7, tol=1e-5)
    return impliedVol

def MertonCallPrice(CP,S0,K,r,tau,muJ,sigmaJ,sigma,xiP):
    X0  = np.log(S0)
    # term for E(exp(J)-1)
    helpExp = np.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0
             
    # Analytical representation for the Merton's option price
    muX     = lambda n: X0 + (r - xiP * helpExp - 0.5 * sigma * sigma) * tau + n * muJ
    sigmaX  = lambda n: np.sqrt(sigma * sigma + n * sigmaJ * sigmaJ / tau) 
    d1      = lambda n: (np.log(S0/K) + (r - xiP * helpExp - 0.5*sigma * sigma \
             + np.power(sigmaX(n),2.0)) * tau + n * muJ) / (sigmaX(n) * np.sqrt(tau))
    d2      = lambda n: d1(n) - sigmaX(n) * np.sqrt(tau)
    value_n = lambda n: np.exp(muX(n) + 0.5*np.power(sigmaX(n),2.0)*tau)\
            * st.norm.cdf(d1(n)) - K *st.norm.cdf(d2(n))
    # Option value calculation, it is infinite sum but we truncate at 20
    valueExact = value_n(0.0)
    kidx = range(1,20)
    for k in kidx:
        valueExact += np.power(xiP * tau, k)*value_n(k) / np.math.factorial(k)
    valueExact *= np.exp(-r*tau) * np.exp(-xiP * tau)
    
    if CP==OptionType.CALL:
        return valueExact
    elif CP==OptionType.PUT:
        return valueExact - S0 + K*np.exp(-r*tau)

def ChFForMertonModel(r,tau,muJ,sigmaJ,sigma,xiP):
    # term for E(exp(J)-1)
    helpExp = np.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0
    
    # Characteristic function for the Merton's model    
    cf = lambda u: np.exp(i * u * (r - xiP * helpExp - 0.5 * sigma * sigma) *tau \
        - 0.5 * sigma * sigma * u * u * tau + xiP * tau * \
        (np.exp(i * u * muJ - 0.5 * sigmaJ * sigmaJ * u * u)-1.0))
    return cf 

def mainCalculation():
    CP  = OptionType.CALL
    S0  = 100
    r   = 0.05
    tau = 10
    
    K = np.linspace(20,5*S0,25)
    K = np.array(K).reshape([len(K),1])
    
    N = 1000
    L = 8
        
    sigma  = 0.15
    muJ    = -0.05
    sigmaJ = 0.3
    xiP    = 0.7
    
    # Evaluate the Merton model
    valueExact = MertonCallPrice(CP,S0,K,r,tau,muJ,sigmaJ,sigma,xiP)
      
    # Compute ChF for th Merton
    cf = ChFForMertonModel(r,tau,muJ,sigmaJ,sigma,xiP)
         
    # The COS method
    valCOS = CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L)
    
    plt.figure(1)
    plt.plot(K,valCOS,'-k')
    plt.xlabel("strike, K")
    plt.ylabel("Option Price")
    plt.grid()    
    plt.plot(K,valueExact,'.r')
    plt.legend(["COS Price","Exact, infinite Sum"])
    print(valCOS)
mainCalculation()