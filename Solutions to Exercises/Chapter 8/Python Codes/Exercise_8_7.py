#%%
"""
Created on Thu Nov 30 2019
Delta, Gamma and Vega for the Heston Model using the COS method
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

def COSMthdDelta(cf,CP,S0,r,tau,K,N,L):
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
    
    # Note additional u * i term for Delta 
    temp = cf(u) * H_k  * u * i
    temp[0] = 0.5 * temp[0]    
    value = 1.0/S0 *np.exp(-r * tau) * K * np.real(mat.dot(temp))     
    return value

def COSMthdGamma(cf,CP,S0,r,tau,K,N,L):
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
    
    # Note additional term for Gamma 
    gamma_term = (u * i)**2.0 - u * i
    temp = cf(u) * H_k  * gamma_term
    temp[0] = 0.5 * temp[0]    
    value = 1.0/(S0**2.0) *np.exp(-r * tau) * K * np.real(mat.dot(temp))     
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
    func = lambda sigma: np.power(BS_Call_Option_Price(CP,S_0,K,sigma,T,r) - marketPrice,1.0)
    impliedVol = optimize.newton(func, 0.7, tol=1e-5)
    return impliedVol

def ChFVegaHestonModel(r,tau,kappa,gamma,vbar,v0,rho):
    i = np.complex(0.0,1.0)
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*np.exp(-D1(u)*tau))) \
        *(kappa-gamma*rho*i*u-D1(u))
    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    A  = lambda u: r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*rho*i*u-D1(u))\
        - 2*kappa*vbar/gamma/gamma*np.log((1-g(u)*np.exp(-D1(u)*tau))/(1-g(u)))
    # Characteristic function for the vega
    cfVega = lambda u: np.exp(A(u) + C(u)*v0) * C(u)
    return cfVega 

def ChFHestonModel(r,tau,kappa,gamma,vbar,v0,rho):
    i = np.complex(0.0,1.0)
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*np.exp(-D1(u)*tau))) \
        *(kappa-gamma*rho*i*u-D1(u))
    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    A  = lambda u: r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*rho*i*u-D1(u))\
        - 2*kappa*vbar/gamma/gamma*np.log((1-g(u)*np.exp(-D1(u)*tau))/(1-g(u)))
    # Characteristic function for the Heston's model    
    cf = lambda u: np.exp(A(u) + C(u)*v0)
    return cf 

def mainCalculation():
    CP  = OptionType.CALL
    S0  = 100
    r   = 0.00
    tau = 1
    
    K = np.linspace(50,150,250)
    K = np.array(K).reshape([len(K),1])
    
    # COS method settings
    L = 10
        
    kappa = 1.5768
    gamma = 0.5751
    vbar  = 0.0398
    rho   =-0.5711
    v0    = 0.0175
    
    # Compute ChF for the Heston model
    cf = ChFHestonModel(r,tau,kappa,gamma,vbar,v0,rho)
         
    # The COS method- REFERENCE
    N = 5000
    callRef = CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L)
    
    plt.figure(1)
    plt.plot(K,callRef,'-k')
    plt.xlabel("strike, K")
    plt.ylabel("Option Price")
    plt.grid()   
    
    # Delta computation- Finite difference
    dSV = np.linspace(10,0.01,10)
    for dS in dSV:
        callUp = CallPutOptionPriceCOSMthd(cf, CP, S0 + dS, r, tau, K, N, L)
        deltaFD = (callUp- callRef) / dS
        
        # Delta computation- COS method 
        deltaCOS = COSMthdDelta(cf,CP,S0,r,tau,K,N,L)
        
        # Error
        max_Error = np.max(np.abs(deltaFD-deltaCOS))
        
        print('Delta Computation: Max ABS Error for {0:.3f} is equal to {1:.5f} '.format(dS,max_Error))
    
    
    plt.figure(2)
    plt.plot(K,deltaFD,'-k')
    plt.plot(K,deltaCOS,'--r')
    plt.xlabel("strike, K")
    plt.ylabel("delta, dCall/dS0")
    plt.legend(['Finite Difference','COS Method'])
    plt.title('Delta')
    plt.grid()   
    
    # Gamma computation- Finite difference
    dSV = np.linspace(10,0.01,10)
    for dS in dSV:
        callUp = CallPutOptionPriceCOSMthd(cf, CP, S0 + dS, r, tau, K, N, L)
        callDown = CallPutOptionPriceCOSMthd(cf, CP, S0 - dS, r, tau, K, N, L)
        gammaFD = (callUp - 2* callRef + callDown ) / (dS**2.0)
        
        # Gamma computation- Finite difference
        gammaCOS = COSMthdGamma(cf,CP,S0,r,tau,K,N,L)
        
        # Error
        max_Error = np.max(np.abs(gammaFD-gammaCOS))
        
        print('Gamma Computation: Max ABS Error for {0:.3f} is equal to {1:.5f} '.format(dS,max_Error))
    
    plt.figure(3)
    plt.plot(K,gammaFD,'-k')
    plt.plot(K,gammaCOS,'--r')
    plt.xlabel("strike, K")
    plt.ylabel("delta, d^2V/dS0^2")
    plt.legend(['Finite Difference','COS Method'])
    plt.title('Gamma')
    plt.grid()   
    
    # Vega 
    dvV = np.linspace(0.01,0.0001,10)
    for dv in dvV:
        cfVegaUp = ChFHestonModel(r,tau,kappa,gamma,vbar,v0 + dv, rho)
        callCOSUp = CallPutOptionPriceCOSMthd(cfVegaUp, CP, S0, r, tau, K, N, L)
        vegaFD = (callCOSUp - callRef) / dv
        
        cfVega = ChFVegaHestonModel(r,tau,kappa,gamma,vbar,v0,rho)
        vegaCOS = CallPutOptionPriceCOSMthd(cfVega, CP, S0, r, tau, K, N, L)    
    
        # Error
        max_Error = np.max(np.abs(vegaFD-vegaCOS))
        
        print('Vega Computation: Max ABS Error for {0:.3f} is equal to {1:.5f} '.format(dv,max_Error))
        
    plt.figure(4)
    plt.plot(K,vegaFD,'-k')
    plt.plot(K,vegaCOS,'--r')
    plt.xlabel("strike, K")
    plt.ylabel("delta, dC/dv0")
    plt.legend(['Finite Difference','COS Method'])
    plt.title('Vega')
    plt.grid() 
    
mainCalculation()