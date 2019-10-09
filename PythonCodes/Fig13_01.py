#%%
"""
Created on Thu Jan 03 2019
The BSHW model and implied volatilities term structure computerion
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as st
import enum 
import scipy.optimize as optimize

# Set i= imaginary number

i   = np.complex(0.0,1.0)

# Time step 

dt = 0.0001
 
# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
def CallPutOptionPriceCOSMthd_StochIR2(cf,CP,S0,tau,K,N,L):


    # cf   - Characteristic function is a function, in the book denoted by \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # tau  - Time to maturity
    # K    - List of strikes
    # N    - Number of expansion terms
    # L    - Size of truncation domain (typ.:L=8 or L=10)
    # P0T  - Zero-coupon bond for maturity T.

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

    H_k = CallPutCoefficients(CP,a,b,k)   
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    value = K * np.real(mat.dot(temp))     
            
    return value

def CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,tau,K,N,L,P0T):


    # cf   - Characteristic function is a function, in the book denoted by \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # tau  - Time to maturity
    # K    - List of strikes
    # N    - Number of expansion terms
    # L    - Size of truncation domain (typ.:L=8 or L=10)
    # P0T  - Zero-coupon bond for maturity T.

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
    value = K * np.real(mat.dot(temp))     
    
    # We use the put-call parity for call options

    if CP == OptionType.CALL:
        value = value + S0 - K * P0T
        
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

def ImpliedVolatilityBlack76(CP,frwdMarketPrice,K,T,frwdStock):
    func = lambda sigma: np.power(BS_Call_Option_Price(CP,frwdStock,K,sigma,T,0.0) - frwdMarketPrice, 1.0)
    impliedVol = optimize.newton(func, 0.2, tol=1e-9)
    #impliedVol = optimize.brent(func, brack= (0.05, 2))
    return impliedVol

def ChFBSHW(u, T, P0T, lambd, eta, rho, sigma):
    i = np.complex(0.0,1.0)
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    
    # Initial interest rate is a forward rate at time t->0

    r0 = f0T(0.00001)
    
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))   
    C = lambda u,tau: 1.0/lambd*(i*u-1.0)*(1.0-np.exp(-lambd*tau))
    
    # Define a grid for the numerical integration of function theta

    zGrid = np.linspace(0.0,T,2500)
    term1 = lambda u: 0.5*sigma*sigma *i*u*(i*u-1.0)*T
    term2 = lambda u: i*u*rho*sigma*eta/lambd*(i*u-1.0)*(T+1.0/lambd *(np.exp(-lambd*T)-1.0))
    term3 = lambda u: eta*eta/(4.0*np.power(lambd,3.0))*np.power(i+u,2.0)*(3.0+np.exp(-2.0*lambd*T)-4.0*np.exp(-lambd*T)-2.0*lambd*T)
    term4 = lambda u:  lambd*integrate.trapz(theta(T-zGrid)*C(u,zGrid), zGrid)    
    A= lambda u: term1(u) + term2(u) + term3(u) + term4(u)
    
    # Note that we don't include B(u)*x0 term as it is already included in the COS method
    cf = lambda u : np.exp(A(u) + C(u,T)*r0 )
    
    # Iterate over all u and collect the ChF, iteration is necessary due to the integration in term4

    cfV = []
    for ui in u:
        cfV.append(cf(ui))
    
    return cfV

def mainCalculation():
    CP  = OptionType.CALL  
        
    # HW model parameter settings

    lambd = 0.1
    eta   = 0.01
    sigma = 0.2
    rho   = 0.3
    S0    = 100
    
    # Strike equals stock value, thus ATM

    K = [100]
    K = np.array(K).reshape([len(K),1])
      
    # We define a ZCB curve (obtained from the market)

    P0T = lambda T: np.exp(-0.05*T) 
    
    # Settings for the COS method

    N = 100
    L = 8
    
    # Maturities at which we compute implied volatility

    TMat = np.linspace(0.1,5.0,20)
        
    # Effect of lambda

    plt.figure(2)
    plt.grid()
    plt.xlabel('maturity, T')
    plt.ylabel('implied volatility')
    lambdV = [0.001,0.1,0.5,1.5]
    legend = []
    for lambdaTemp in lambdV:    
       IV =np.zeros([len(TMat),1])
       for idx in range(0,len(TMat)):
           T = TMat[idx]

           # Compute ChF for the BSHW model     

           cf = lambda u: ChFBSHW(u, T, P0T, lambdaTemp, eta, rho, sigma)
           valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
           frwdStock = S0 / P0T(T)
           valCOSFrwd = valCOS / P0T(T)
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd,K,T,frwdStock)
       plt.plot(TMat,IV*100.0)       
       legend.append('lambda={0}'.format(lambdaTemp))
    plt.legend(legend)    

    # Effect of eta

    plt.figure(3)
    plt.grid()
    plt.xlabel('maturity, T')
    plt.ylabel('implied volatility')
    etaV = [0.001,0.05,0.1,0.15]
    legend = []
    for etaTemp in etaV:    
       IV =np.zeros([len(TMat),1])
       for idx in range(0,len(TMat)):
           T = TMat[idx]
           cf = lambda u: ChFBSHW(u, T, P0T, lambd, etaTemp, rho, sigma)
           valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
           frwdStock = S0 / P0T(T)
           valCOSFrwd = valCOS/P0T(T)
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd,K,T,frwdStock)
       plt.plot(TMat,IV*100.0)       
       legend.append('eta={0}'.format(etaTemp))
    plt.legend(legend)    
    
    # Effect of sigma

    plt.figure(4)
    plt.grid()
    plt.xlabel('maturity, T')
    plt.ylabel('implied volatility')
    sigmaV = [0.1,0.2,0.3,0.4]
    legend = []
    for sigmaTemp in sigmaV:    
       IV =np.zeros([len(TMat),1])
       for idx in range(0,len(TMat)):
           T = TMat[idx]
           cf = lambda u: ChFBSHW(u, T, P0T, lambd, eta, rho, sigmaTemp)
           valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
           frwdStock = S0 / P0T(T)
           valCOSFrwd = valCOS / P0T(T)
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd,K,T,frwdStock)
       plt.plot(TMat,IV*100.0)       
       legend.append('sigma={0}'.format(sigmaTemp))
    plt.legend(legend)    

    # Effect of rho

    plt.figure(5)
    plt.grid()
    plt.xlabel('maturity, T')
    plt.ylabel('implied volatility')
    rhoV = [-0.7, -0.3, 0.3, 0.7]
    legend = []
    for rhoTemp in rhoV:    
       IV =np.zeros([len(TMat),1])
       for idx in range(0,len(TMat)):
           T = TMat[idx]
           cf = lambda u: ChFBSHW(u, T, P0T, lambd, eta, rhoTemp, sigma)
           valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
           frwdStock = S0 / P0T(T)
           valCOSFrwd = valCOS / P0T(T)
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd,K,T,frwdStock)
       plt.plot(TMat,IV*100.0)       
       legend.append('rho={0}'.format(rhoTemp))
    plt.legend(legend) 

mainCalculation()
