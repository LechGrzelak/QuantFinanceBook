#%%
"""
Created on Thu Jan 03 2019
The SZHW model and implied volatilities 
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
    
    #r Aassigning i=sqrt(-1)

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
    N = 500
    arg=np.linspace(0,tau,N)

    for k in range(0,len(u)):
       E_val=E(u[k],arg,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar)
       C_val=C(u[k],arg,lambd)
       f=(kappa*sigmabar+1.0/2.0*gamma*gamma*E_val+gamma*eta*Rrsigma*C_val)*E_val
       value1 =integrate.trapz(np.real(f),arg)
       value2 =integrate.trapz(np.imag(f),arg)
       value[k]=(value1 + value2*i)#np.complex(value1,value2)
    
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

def ChFBSHW(u, T, P0T, lambd, eta, rho, sigma):
    i = np.complex(0.0,1.0)
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    
    # Initial interest rate is forward rate at time t->0

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
    
    # Note that we don't include the B(u)*x0 term as it is in the COS method

    cf = lambda u : np.exp(A(u) + C(u,T)*r0 )
    
    # Iterate over all u and collect the ChF, iteration is necessary due to the integration in term4

    cfV = []
    for ui in u:
        cfV.append(cf(ui))
    
    return cfV

def mainCalculation():
    CP  = OptionType.CALL  
        
    # HW model parameter settings

    lambd = 0.425
    eta   = 0.1
    S0    = 100
    T     = 5.0
    
    # The SZHW model

    sigma0  = 0.1 
    gamma   = 0.11 
    Rrsigma = 0.32
    Rxsigma = -0.42
    Rxr     = 0.3
    kappa   = 0.4
    sigmabar= 0.05
    
    # Strike range

    K = np.linspace(40,200.0,20)
    K = np.array(K).reshape([len(K),1])
      
    # We define a ZCB curve (obtained from the market)

    P0T = lambda T: np.exp(-0.025*T) 

    # Forward stock

    frwdStock = S0 / P0T(T)
    
    # Settings for the COS method

    N = 2000
    L = 10   
 
    # Effect of gamma

    plt.figure(1)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    gammaV = [0.1, 0.2, 0.3, 0.4]
    legend = []
    for gammaTemp in gammaV:    

        # Evaluate the SZHW model

       cf = lambda u: ChFSZHW(u,P0T,sigma0,T,lambd,gammaTemp,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar)
       #cf = lambda u: ChFBSHW(u, T, P0T, lambd, eta,  -0.7, 0.1)

       # The COS method

       valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
       valCOSFrwd = valCOS/P0T(T)

       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd[idx],K[idx],T,frwdStock)
       plt.plot(K,IV*100.0)
       legend.append('gamma={0}'.format(gammaTemp))
    plt.legend(legend)
   
    
    # Effect of kappa

    plt.figure(2)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    kappaV = [0.05, 0.2, 0.3, 0.4]
    legend = []
    for kappaTemp in kappaV:    

        # Evaluate the SZHW model

       cf = lambda u: ChFSZHW(u,P0T,sigma0,T,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappaTemp,sigmabar)
       
       # The COS method

       valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
       valCOSFrwd = valCOS/P0T(T)

       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd[idx],K[idx],T,frwdStock)
       plt.plot(K,IV*100.0)
       legend.append('kappa={0}'.format(kappaTemp))
    plt.legend(legend)    
    
   
    # Effect of rhoxsigma

    plt.figure(3)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    RxsigmaV = [-0.75, -0.25, 0.25, 0.75]
    legend = []
    for RxsigmaTemp in RxsigmaV:    

        # Evaluate the SZHW model

       cf = lambda u: ChFSZHW(u,P0T,sigma0,T,lambd,gamma,RxsigmaTemp,Rrsigma,Rxr,eta,kappa,sigmabar)
       
       # The COS method

       valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
       valCOSFrwd = valCOS/P0T(T)

       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd[idx],K[idx],T,frwdStock)
       plt.plot(K,IV*100.0)
       legend.append('Rxsigma={0}'.format(RxsigmaTemp))
    plt.legend(legend)    
    
    #Effect of sigmabar

    plt.figure(4)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    sigmabarV = [0.1, 0.2, 0.3, 0.4]
    legend = []
    for sigmabarTemp in sigmabarV:    

        # Evaluate the SZHW model

       cf = lambda u: ChFSZHW(u,P0T,sigma0,T,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabarTemp)
       
       # The COS method

       valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
       valCOSFrwd = valCOS/P0T(T)

       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd[idx],K[idx],T,frwdStock)
       plt.plot(K,IV*100.0)
       legend.append('sigmabar={0}'.format(sigmabarTemp))
    plt.legend(legend)  
    
mainCalculation()
