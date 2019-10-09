#%%
"""
Created on Feb 11 2019
Local volatility model based on the implied volatility obtained from prices from the Heston model
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as optimize
import enum 

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
def CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L):

    # cf   - Characteristic function, in the book denoted by \varphi
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

    H_k = CallPutCoefficients(CP,a,b,k)   
    mat = np.exp(i * np.outer((x0 - a) , u))
    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))     
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

def BS_Call_Put_Option_Price(CP,S_0,K,sigma,tau,r):
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * tau) / (sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

# Implied volatility method

def ImpliedVolatility(CP,marketPrice,K,T,S_0,r):

    # To determine initial volatility we define a grid for sigma
    # and interpolate on the inverse function

    sigmaGrid = np.linspace(0,2,200)
    optPriceGrid = BS_Call_Put_Option_Price(CP,S_0,K,sigmaGrid,T,r)
    sigmaInitial = np.interp(marketPrice,optPriceGrid,sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))
    
    # Use already determined input for the local-search (final tuning)

    func = lambda sigma: np.power(BS_Call_Put_Option_Price(CP,S_0,K,sigma,T,r) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-15)
    print("Final volatility = {0}".format(impliedVol))
    return impliedVol

def ChFHestonModel(r,tau,kappa,gamma,vbar,v0,rho):
    i = np.complex(0.0,1.0)
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*np.exp(-D1(u)*tau)))\
        *(kappa-gamma*rho*i*u-D1(u))

    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method

    A  = lambda u: r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*rho*i*u-D1(u))\
        - 2*kappa*vbar/gamma/gamma*np.log((1.0-g(u)*np.exp(-D1(u)*tau))/(1.0-g(u)))

    # Characteristic function for the Heston model    

    cf = lambda u: np.exp(A(u) + C(u)*v0)
    return cf 

def OptionPriceWithCosMethodHelp(CP,T,K,S0,r,kappa,gamma,v0,vbar,rho):

     # Settings for the COS method

    N = 500
    L = 8
           
    # ChF for the Heston model

    cf = ChFHestonModel(r,T,kappa,gamma,vbar,v0,rho)
    OptPrice = CallPutOptionPriceCOSMthd(cf, CP, S0, r, T, K, N, L)   
    return OptPrice
    
def EUOptionPriceFromMCPaths(CP,S,K,T,r):

    # S is a vector of Monte Carlo samples at T

    if CP == OptionType.CALL:
        return np.exp(-r*T)*np.mean(np.maximum(S-K,0.0))
    elif CP == OptionType.PUT:
        return np.exp(-r*T)*np.mean(np.maximum(K-S,0.0))
    
def mainCalculation():

    # Heston model parameters   

    kappa   = 1.3
    vbar    = 0.05
    gamma   = 0.3 
    rho     = -0.3 
    v0      = 0.1
    r       = 0.06    
    S0     = 1.0
    T      = 1.0
    r      = 0.06
    CP     = OptionType.CALL

    # Monte Carlo settings

    NoOfPaths = 5000
    NoOfSteps = (int)(250*T)
    
       
    # Define a function to calculate option prices

    V =lambda T,K: OptionPriceWithCosMethodHelp(CP,T,K,S0,r,kappa,gamma,v0,vbar,rho)
   
    # Define a shock size for derivative calculation

    bump_T  = 1e-4
    bump_K  = 1e-4
    
    # Define derivatives

    dV_dT   = lambda T,K: (V(T + bump_T,K) - V(T ,K)) / bump_T
    dV_dK   = lambda T,K: (V(T,K + bump_K) - V(T,K - bump_K)) / (2.0 * bump_K)
    d2V_dK2 = lambda T,K: (V(T,K + bump_K) + V(T,K-bump_K) - 2.0*V(T,K))/(bump_K**2.0 )

    # Local variance

    sigmaLV2=lambda T,K: (dV_dT(T,K) + r * K * dV_dK(T,K)) / (0.5 * K**2.0 * d2V_dK2(T,K))

    # Monte Carlo simulation

    dt        = T/NoOfSteps
    np.random.seed(5)
    Z         = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    S         = np.zeros([NoOfPaths,NoOfSteps+1])
    
    S[:,0]    = S0;
    time = np.zeros([NoOfSteps+1,1])
    
    for i in range(0,NoOfSteps):

        # This condition is necessary as for t=0 we cannot compute implied
        # volatilities

        if time[i]==0.0:
            time[i]=0.0001
        
        print('current time is {0}'.format(time[i]))

        # Standarize Normal(0,1)

        Z[:,i]=(Z[:,i]-np.mean(Z[:,i]))/np.std(Z[:,i])
        
        # Compute local volatility

        S_i = np.array(S[:,i]).reshape([NoOfPaths,1])
        
        if time[i]  >0 :
            sig = np.real(sigmaLV2(time[i],S_i))
        elif time[i]==0:
            sig = np.real(sigmaLV2(dt/2.0,S_i))
        
        np.nan_to_num(sig)
        
        # Because of discretizations we may encouter negative variance which
        # are set to 0 here.

        sig=np.maximum(sig,1e-14)
        sigmaLV = np.sqrt(sig.transpose())  
        
        # Stock path

        S[:,i+1] = S[:,i] * (1.0 + r*dt + sigmaLV*Z[:,i]*np.sqrt(dt))
                       
        # We force that at each time S(t)/M(t) is a martingale

        S[:,i+1] = S[:,i+1] - np.mean(S[:,i+1]) + S0*np.exp(r*(time[i]+dt))
        
        # Make sure that after moment matching we don't encounter negative stock values

        S[:,i+1] = np.maximum(S[:,i+1],1e-14)
        
        # Adjust time

        time[i+1] = time[i] + dt
        
    # Plot some results

    K = np.linspace(0.5,1.7,50)
    OptPrice = np.zeros([len(K),1])
    IV_Heston = np.zeros([len(K),1])
    IV_MC = np.zeros([len(K),1])
    
    # Prices from the Heston model

    valCOS = V(T,K)
    
    # Implied volatilities

    for (idx,k) in enumerate(K):
        OptPrice[idx] = EUOptionPriceFromMCPaths(CP,S[:,-1],k,T,r)
        IV_MC[idx]    = ImpliedVolatility(CP,OptPrice[idx],k,T,S0,r)*100.0
        IV_Heston[idx] =ImpliedVolatility(CP,valCOS[idx],K[idx],T,S0,r)*100
        
    # Plot the option prices

    plt.figure(1)
    plt.plot(K,OptPrice)
    plt.grid()
    plt.xlabel('strike')
    plt.ylabel('option price')
    
    # Plot the implied volatilities

    plt.figure(2)
    plt.plot(K,IV_Heston)
    plt.plot(K,IV_MC,'-r')
    plt.grid()
    plt.xlabel('strike')
    plt.ylabel('implied volatility')
    plt.legend(['Heston','Monte Carlo'])
    plt.axis([np.min(K),np.max(K),0,40])
    plt.ylim([25,35])
mainCalculation()
