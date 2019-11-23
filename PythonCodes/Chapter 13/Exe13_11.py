#%%
"""
Created on Thu Jan 03 2019
The SZHW model and implied volatilities and comparison to Monte Carlo 
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

# Time step needed for differentiation

dt = 0.0001
 
# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
def CallPutOptionPriceCOSMthd_StochIR(cf,CP,S0,tau,K,N,L,P0T):


    # cf   - Characteristic function, in the book denoted by \varphi
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
    u = k * np.pi / (b - a)  

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

def GeneratePathsSZHWEuler(NoOfPaths,NoOfSteps,P0T,T,S0,sigma0,sigmabar,kappa,gamma,lambd,eta,Rxsigma,Rxr,Rsigmar):    

    # Time step needed for differentiation

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    
    # Initial interest rate is forward rate at time t->0

    r0 = f0T(0.00001)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))      
    
    # Empty containers for the Brownian motion

    Wx = np.zeros([NoOfPaths, NoOfSteps+1])
    Wsigma = np.zeros([NoOfPaths, NoOfSteps+1])
    Wr = np.zeros([NoOfPaths, NoOfSteps+1])
    
    Sigma = np.zeros([NoOfPaths, NoOfSteps+1])
    X     = np.zeros([NoOfPaths, NoOfSteps+1])
    R     = np.zeros([NoOfPaths, NoOfSteps+1])
    M_t   = np.ones([NoOfPaths,NoOfSteps+1])
    R[:,0]     = r0 
    Sigma[:,0] = sigma0
    X[:,0]     = np.log(S0)
    
    dt = T / float(NoOfSteps)    
    cov = np.array([[1.0, Rxsigma,Rxr],[Rxsigma,1.0,Rsigmar], [Rxr,Rsigmar,1.0]])
      
    time = np.zeros([NoOfSteps+1])    

    for i in range(0,NoOfSteps):
        Z = np.random.multivariate_normal([.0,.0,.0],cov,NoOfPaths)
        if NoOfPaths > 1:
            Z[:,0] = (Z[:,0] - np.mean(Z[:,0])) / np.std(Z[:,0])
            Z[:,1] = (Z[:,1] - np.mean(Z[:,1])) / np.std(Z[:,1])
            Z[:,2] = (Z[:,2] - np.mean(Z[:,2])) / np.std(Z[:,2])
            
        Wx[:,i+1]     = Wx[:,i]     + np.power(dt, 0.5)*Z[:,0]
        Wsigma[:,i+1] = Wsigma[:,i] + np.power(dt, 0.5)*Z[:,1]
        Wr[:,i+1]     = Wr[:,i]     + np.power(dt, 0.5)*Z[:,2]

        # Euler discretization

        R[:,i+1]     = R[:,i] + lambd*(theta(time[i]) - R[:,i])*dt + eta * (Wr[:,i+1]-Wr[:,i])
        M_t[:,i+1]   = M_t[:,i] * np.exp(0.5*(R[:,i+1] + R[:,i])*dt)
        Sigma[:,i+1] = Sigma[:,i] + kappa*(sigmabar - Sigma[:,i])*dt + gamma* (Wsigma[:,i+1]-Wsigma[:,i])                        
        X[:,i+1]     = X[:,i] + (R[:,i] - 0.5*Sigma[:,i]**2.0)*dt + Sigma[:,i] * (Wx[:,i+1]-Wx[:,i])                        
        time[i+1] = time[i] +dt
        
    paths = {"time":time,"S":np.exp(X),"M_t":M_t}
    return paths

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

def ImpliedVolatilityBlack76(CP,marketPrice,K,T,S_0):

    # To determine the initial volatility we define a grid for sigma
    # and interpolate on the inverse function

    sigmaGrid = np.linspace(0.0,5.0,5000)
    optPriceGrid = BS_Call_Put_Option_Price(CP,S_0,K,sigmaGrid,T,0.0)
    sigmaInitial = np.interp(marketPrice,optPriceGrid,sigmaGrid)
    print("Strike = {0}".format(K))
    print("Initial volatility = {0}".format(sigmaInitial))
    
    # Use already determined input for the local-search (final tuning)

    func = lambda sigma: np.power(BS_Call_Put_Option_Price(CP,S_0,K,sigma,T,0.0) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-15)
    print("Final volatility = {0}".format(impliedVol))
    if impliedVol > 2.0:
        impliedVol = 0.0
    return impliedVol

def C(u,tau,lambd):
    i     = complex(0,1)
    return 1.0/lambd*(i*u-1.0)*(1.0-np.exp(-lambd*tau))

def D(u,tau,kappa,Rxsigma,gamma):
    i=np.complex(0.0,1.0)
    a_0=-1.0/2.0*u*(i+u)
    a_1=2.0*(gamma*Rxsigma*i*u-kappa)
    a_2=2.0*gamma*gamma
    d=np.sqrt(a_1*a_1-4.0*a_0*a_2)
    g=(-a_1-d)/(-a_1+d)    
    return (-a_1-d)/(2.0*a_2*(1.0-g*np.exp(-d*tau)))*(1.0-np.exp(-d*tau))
    
def E(u,tau,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar):
    i=np.complex(0.0,1.0)
    a_0=-1.0/2.0*u*(i+u)
    a_1=2.0*(gamma*Rxsigma*i*u-kappa)
    a_2=2*gamma*gamma
    d  =np.sqrt(a_1*a_1-4.0*a_0*a_2)
    g =(-a_1-d)/(-a_1+d)    
    
    c_1=gamma*Rxsigma*i*u-kappa-1.0/2.0*(a_1+d)
    f_1=1.0/c_1*(1.0-np.exp(-c_1*tau))+1.0/(c_1+d)*(np.exp(-(c_1+d)*tau)-1.0)
    f_2=1.0/c_1*(1.0-np.exp(-c_1*tau))+1.0/(c_1+lambd)*(np.exp(-(c_1+lambd)*tau)-1.0)
    f_3=(np.exp(-(c_1+d)*tau)-1.0)/(c_1+d)+(1.0-np.exp(-(c_1+d+lambd)*tau))/(c_1+d+lambd)
    f_4=1.0/c_1-1.0/(c_1+d)-1.0/(c_1+lambd)+1.0/(c_1+d+lambd)
    f_5=np.exp(-(c_1+d+lambd)*tau)*(np.exp(lambd*tau)*(1.0/(c_1+d)-np.exp(d*tau)/c_1)+np.exp(d*tau)/(c_1+lambd)-1.0/(c_1+d+lambd)) 

    I_1=kappa*sigmabar/a_2*(-a_1-d)*f_1
    I_2=eta*Rxr*i*u*(i*u-1.0)/lambd*(f_2+g*f_3)
    I_3=-Rrsigma*eta*gamma/(lambd*a_2)*(a_1+d)*(i*u-1)*(f_4+f_5)
    return np.exp(c_1*tau)*1.0/(1.0-g*np.exp(-d*tau))*(I_1+I_2+I_3)

def A(u,tau,eta,lambd,Rxsigma,Rrsigma,Rxr,gamma,kappa,sigmabar):
    i=np.complex(0.0,1.0)
    a_0=-1.0/2.0*u*(i+u)
    a_1=2.0*(gamma*Rxsigma*i*u-kappa)
    a_2=2.0*gamma*gamma
    d  =np.sqrt(a_1*a_1-4.0*a_0*a_2)
    g =(-a_1-d)/(-a_1+d) 
    f_6=eta*eta/(4.0*np.power(lambd,3.0))*np.power(i+u,2.0)*(3.0+np.exp(-2.0*lambd*tau)-4.0*np.exp(-lambd*tau)-2.0*lambd*tau)
    A_1=1.0/4.0*((-a_1-d)*tau-2.0*np.log((1-g*np.exp(-d*tau))/(1.0-g)))+f_6
  
    # Integration within the function A(u,tau)

    value=np.zeros([len(u),1],dtype=np.complex_)   
    N = 100
    z1 = np.linspace(0,tau,N)
    #arg =z1

    E_val=lambda z1,u: E(u,z1,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar)
    C_val=lambda z1,u: C(u,z1,lambd)
    f    =lambda z1,u: (kappa*sigmabar+1.0/2.0*gamma*gamma*E_val(z1,u)+gamma*eta*Rrsigma*C_val(z1,u))*E_val(z1,u)
    
    value1 =integrate.trapz(np.real(f(z1,u)),z1).reshape(u.size,1)
    value2 =integrate.trapz(np.imag(f(z1,u)),z1).reshape(u.size,1)
    value  =(value1 + value2*i)
   
    return value + A_1

def ChFSZHW(u,P0T,sigma0,tau,lambd,gamma,    Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar):
    v_D = D(u,tau,kappa,Rxsigma,gamma)
    v_E = E(u,tau,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar)
    v_A = A(u,tau,eta,lambd,Rxsigma,Rrsigma,Rxr,gamma,kappa,sigmabar)
    
    v_0 = sigma0*sigma0
    
    hlp = eta*eta/(2.0*lambd*lambd)*(tau+2.0/lambd*(np.exp(-lambd*tau)-1.0)-1.0/(2.0*lambd)*(np.exp(-2.0*lambd*tau)-1.0))
      
    correction = (i*u-1.0)*(np.log(1/P0T(tau))+hlp)
          
    cf = np.exp(v_0*v_D + sigma0*v_E + v_A + correction)
    return cf.tolist()

def EUOptionPriceFromMCPathsGeneralizedStochIR(CP,S,K,T,M):

    # S is a vector of Monte Carlo samples at T

    result = np.zeros([len(K),1])
    if CP == OptionType.CALL:
        for (idx,k) in enumerate(K):
            result[idx] = np.mean(1.0/M*np.maximum(S-k,0.0))
    elif CP == OptionType.PUT:
        for (idx,k) in enumerate(K):
            result[idx] = np.mean(1.0/M*np.maximum(k-S,0.0))
    return result


def mainCalculation():
    CP  = OptionType.CALL  
        
    # HW model parameter settings

    lambd = 0.05
    eta   = 0.01
    S0    = 100.0
    T     = 5.0
    
    # The SZHW model

    sigma0  = 0.1 
    gamma   = 0.3 
    Rsigmar = 0.32
    Rxsigma = -0.42
    Rxr     = 0.3
    kappa   = 0.4
    sigmabar= 0.05
    
    # Monte Carlo setting

    NoOfPaths =10000
    NoOfSteps = int(100*T)
    
    # We define a ZCB curve (obtained from the market)

    P0T = lambda T: np.exp(-0.025*T) 
    
    # Forward stock

    frwdStock = S0 / P0T(T)
    
    # Strike range

    K = np.linspace(0.4*frwdStock,1.6*frwdStock,20)
    K = np.array(K).reshape([len(K),1])
      
    # Settings for the COS method

    N = 2000
    L = 10   
 
    # Number of repeated simulations for different Monte Carlo seeds

    SeedV = range(0,20) 
    optMCM = np.zeros([len(SeedV),len(K)])
    for (idx,seed) in enumerate(SeedV):
        print('Seed number = {0} out of {1}'.format(idx,len(SeedV)))
        np.random.seed(seed)
        paths = GeneratePathsSZHWEuler(NoOfPaths,NoOfSteps,P0T,T,S0,sigma0,sigmabar,kappa,gamma,lambd,eta,Rxsigma,Rxr,Rsigmar)
        S = paths["S"]
        M_t = paths["M_t"]
        optMC = EUOptionPriceFromMCPathsGeneralizedStochIR(CP,S[:,-1],K,T,M_t[:,-1])
        optMCM[idx,:]= np.squeeze(optMC)
    
    # Average of the runs + standard deviation

    optionMC_E = np.zeros([len(K)])
    for (idx,k) in enumerate(K):
        optionMC_E[idx] = np.mean(optMCM[:,idx])
        
    print('Martinagle property check={0}',np.mean(S[:,-1]/M_t[:,-1]))

    # Evaluate the SZHW model with the COS method

    cf = lambda u: ChFSZHW(u,P0T,sigma0,T,lambd,gamma,Rxsigma,Rsigmar,Rxr,eta,kappa,sigmabar)

    # The COS method

    valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
    
    # Option prices

    plt.figure(1)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('Option Prices')
    plt.plot(K,valCOS)
    plt.plot(K,optionMC_E,'--r')
    plt.legend(['COS method','Monte Carlo'])
    
    # Implied volatilities

    valCOSFrwd = valCOS/P0T(T)
    valMCFrwd = optionMC_E/P0T(T)
    IVCOS =np.zeros([len(K),1])
    IVMC =np.zeros([len(K),1])
    for idx in range(0,len(K)):
        IVCOS[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd[idx],K[idx],T,frwdStock)*100.0
        IVMC[idx] = ImpliedVolatilityBlack76(CP,valMCFrwd[idx],K[idx],T,frwdStock)*100.0
        
    #  Implied Volatilities

    plt.figure(2)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    plt.plot(K,IVCOS)
    plt.plot(K,IVMC,'--r')
    plt.title('Implied volatilities')
    plt.legend(['IV-COS','IV-MC'])
    
    
mainCalculation()
