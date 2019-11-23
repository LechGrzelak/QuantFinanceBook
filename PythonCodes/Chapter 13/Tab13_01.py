#%%
"""
Created on Thu Jan 03 2019
The SZHW model and implied volatilities, the calibration takes place for 
a single maturity. The numbers reported in the table may differ with respect to 
function outputs. Here we assume that the market option prices are given by the
Heston model while in the book we consider real market data.
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as st
import enum 
import scipy.optimize as optimize
from scipy.optimize import minimize

# Set i= imaginary number

i   = np.complex(0.0,1.0)

# Time step

dt = 0.0001
 
# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0
    
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

    """value=np.zeros([len(u),1],dtype=np.complex_)   
    N = 25
    arg=np.linspace(0,tau,N)

    for k in range(0,len(u)):
       E_val=E(u[k],arg,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar)
       C_val=C(u[k],arg,lambd)
       f=(kappa*sigmabar+1.0/2.0*gamma*gamma*E_val+gamma*eta*Rrsigma*C_val)*E_val
       value1 =integrate.trapz(np.real(f),arg)
       value2 =integrate.trapz(np.imag(f),arg)
       value[k]=(value1 + value2*i)#np.complex(value1,value2)
    """

    # Integration within the function A(u,tau)

    value=np.zeros([len(u),1],dtype=np.complex_)   
    N = 25
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

def ChFHestonModelFrwd(r,tau,kappa,gamma,vbar,v0,rho):
    i = np.complex(0.0,1.0)
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*np.exp(-D1(u)*tau)))\
        *(kappa-gamma*rho*i*u-D1(u))

    # Note that we exclude the term -r*tau, as discounting is performed in the COS method

    A  = lambda u: -r*tau + r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*rho*i*u-D1(u))\
        - 2*kappa*vbar/gamma/gamma*np.log((1.0-g(u)*np.exp(-D1(u)*tau))/(1.0-g(u)))

    # Characteristic function for the Heston model    

    cf = lambda u: np.exp(A(u) + C(u)*v0)
    return cf

# Calibration of the SZHW model 

def calibrationSZHW_Global(CP,kappa,Rxr,eta,lambd,K,marketPrice,S0,T,P0T):
    K = np.array(K)
    marketPrice = np.array(marketPrice)
    # x = [gamma,sigmaBar,Rrsigma,Rxsigma,sigma0]
    f_obj = lambda x: TargetVal(CP,kappa,x[0],x[1],Rxr,x[2],x[3],x[4],eta,lambd,K,marketPrice,S0,T,P0T)
    
    # Random initial guess

    #[gamma,sigmaBar,Rrsigma,Rxsigma,sigma0
    initial = np.array([0.1, 0.3,0.0,-0.5,0.3])
    
    # The bounds

    xmin = [0.1, 0.01,-0.85, -0.85, 0.001]
    xmax = [0.8,  0.6, 0.85, -0.0, 0.8]    
    
    # Rewrite the bounds as required for L-BFGS-B

    bounds = [(low, high) for low, high in zip(xmin, xmax)]

    # Use L-BFGS-B method because the problem is smooth and bounded

    minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds,options={'niter':1})

    # Global search

    pars = optimize.basinhopping(f_obj, initial,niter=1, minimizer_kwargs=minimizer_kwargs)
    
    gamma_est = pars.x[0]
    sigmaBar_est = pars.x[1]
    Rrsigma_est = pars.x[2]
    Rxsigma_est = pars.x[3]
    sigma0_est = pars.x[4]
    print("=======================Start local search!========================")
    print(pars)

    # Use global parameters in the local search

    initial =[gamma_est,sigmaBar_est,Rrsigma_est,Rxsigma_est,sigma0_est] 
    pars  = minimize(f_obj,initial,method='nelder-mead', options = \
                     {'xtol': 1e-05, 'disp': False,'maxiter':20})
    print(pars)
    
    gamma_est = pars.x[0]
    sigmaBar_est = pars.x[1]
    Rrsigma_est = pars.x[2]
    Rxsigma_est = pars.x[3]
    sigma0_est = pars.x[4]
    parmCalibr =  {"gamma":gamma_est,"sigmaBar":sigmaBar_est,"Rrsigma":Rrsigma_est,\
                   "Rxsigma":Rxsigma_est,"sigma0":sigma0_est,'ErrorFinal':pars.fun}
    return parmCalibr

# Calibration of the SZHW model 

def calibrationSZHW(CP,kappa,Rxr,eta,lambd,K,marketPrice,S0,T,P0T):
    K = np.array(K)
    marketPrice = np.array(marketPrice)
    # x = [gamma,sigmaBar,Rrsigma,Rxsigma,sigma0]
    f_obj = lambda x: TargetVal(CP,kappa,x[0],x[1],Rxr,x[2],x[3],x[4],eta,lambd,K,marketPrice,S0,T,P0T)
    
    # Random initial guess

    #[gamma,sigmaBar,Rrsigma,Rxsigma,sigma0
    initial = np.array([0.1, 0.1, 0.0,-0.5,0.1])
    pars  = minimize(f_obj,initial,method='nelder-mead', options = \
                     {'xtol': 1e-05, 'disp': False,'maxiter':100})
    print(pars)
    
    gamma_est = pars.x[0]
    sigmaBar_est = pars.x[1]
    Rrsigma_est = pars.x[2]
    Rxsigma_est = pars.x[3]
    sigma0_est = pars.x[4]
    parmCalibr =  {"gamma":gamma_est,"sigmaBar":sigmaBar_est,"Rrsigma":Rrsigma_est,\
                   "Rxsigma":Rxsigma_est,"sigma0":sigma0_est}
    return parmCalibr

def TargetVal(CP,kappa,gamma,sigmaBar,Rxr,Rrsigma,Rxsigma,sigma0,eta,lambd,K,marketPrice,S0,T,P0T):
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    # Settings for the COS method

    N = 1000
    L = 15  
    cf = lambda u: ChFSZHW(u,P0T,sigma0,T,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmaBar)
    valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
    
    # Error is defined as the difference between the market and the model

    errorVector = valCOS - marketPrice
    
    # Target value is a norm of the error vector

    value       = np.linalg.norm(errorVector)   
    print("Total Error = {0}".format(value))
    return value


def mainCalculation():
    CP  = OptionType.CALL  
        
    # HW model parameter settings

    lambd = 1.12
    eta   = 0.02
    S0    = 100.0
    T     = 10.0       
    r     = 0.033
   
    # Fixed SZHW model parameters

    kappa =  0.5
    Rxr   =  0.8
    
    # Strike range

    K = np.linspace(50.0,200.0,20)
    K = np.array(K).reshape([len(K),1])
      
    # We define a ZCB curve (obtained from the market)

    P0T = lambda T: np.exp(-r*T) 
       
    # Settings for the COS method

    N = 2000
    L = 25  
    
    ################## Here we define market call option prices #################
    # We assume, in this experiment reference option prices by the Heston model
    # Note that for the Heston model we deal with the forward ChF, i.e. we have 
    # additional r*tau term in the exponent

    # ChF for the Heston model

    kappaH = 0.5
    vbarH  = 0.0770
    v0H    = 0.0107
    rhoH   = -0.6622
    gammaH = 0.35
    cfHeston = ChFHestonModelFrwd(r,T,kappaH,gammaH,vbarH,v0H,rhoH)
    referencePrice = CallPutOptionPriceCOSMthd_StochIR(cfHeston, CP, S0, T, K, N, L,P0T(T))    
    
    plt.figure(1)
    plt.plot(K,referencePrice)
    plt.xlabel('strike, K')
    plt.ylabel('reference option price')
    plt.grid()
    
    # Calibrate the model and show the output

    calibratedParms = calibrationSZHW_Global(CP,kappa,Rxr,eta,lambd,K,referencePrice,S0,T,P0T)
    
    # Plot calibrated SZHW prices          

    gamma    =calibratedParms.get('gamma')
    sigmaBar =calibratedParms.get('sigmaBar')
    Rrsigma = calibratedParms.get('Rrsigma')
    Rxsigma= calibratedParms.get('Rxsigma')
    sigma0= calibratedParms.get('sigma0')   
    cf = lambda u: ChFSZHW(u,P0T,sigma0,T,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmaBar)
    valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
    plt.plot(K,valCOS,'--r')
    
    print("Optimal parameters are: gamma = {0:.3f}, sigmaBar = {1:.3f}, Rrsigma \
          = {2:.3f}, Rxsigma = {3:.3f}, sigma0 = {4:.3f}".format(gamma,\
          sigmaBar,Rrsigma,Rxsigma,sigma0))
    
mainCalculation()
