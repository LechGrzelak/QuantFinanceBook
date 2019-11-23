#%%
"""
Created on Feb 27 2019
The SZHW and the H1-HW models with the calibration that takes place for 
a single maturity. 
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as st
import scipy.special as sp
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
    if sigma is list:
        sigma = np.array(sigma).reshape([len(sigma),1])
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

    # To determine initial volatility we define a grid for sigma
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
    
    """
    for k in range(0,len(u)):
       E_val=E(u[k],arg,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmabar)
       C_val=C(u[k],arg,lambd)
       f=(kappa*sigmabar+1.0/2.0*gamma*gamma*E_val+gamma*eta*Rrsigma*C_val)*E_val
       value1 =integrate.trapz(np.real(f),arg)
       value2 =integrate.trapz(np.imag(f),arg)
       value[k]=(value1 + value2*i)
    """
    
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

# Exact expectation E(sqrt(V(t)))

def meanSqrtV_3(kappa,v0,vbar,gamma):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= lambda t: 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t)))
    kappaBar = lambda t: 4.0*kappa*v0*np.exp(-kappa*t)/(gamma*gamma*(1.0-np.exp(-kappa*t)))
    temp1 = lambda t: np.sqrt(2.0*c(t))* sp.gamma((1.0+delta)/2.0)/sp.gamma(delta/2.0)*sp.hyp1f1(-0.5,delta/2.0,-kappaBar(t)/2.0)
    return temp1

def C_H1HW(u,tau,lambd):
    i = np.complex(0.0,1.0)
    C = (i*u - 1.0)/lambd * (1-np.exp(-lambd*tau))
    return C

def D_H1HW(u,tau,kappa,gamma,rhoxv):
    i = np.complex(0.0,1.0)
    
    D1 = np.sqrt(np.power(kappa-gamma*rhoxv*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = (kappa-gamma*rhoxv*i*u-D1)/(kappa-gamma*rhoxv*i*u+D1)
    D  = (1.0-np.exp(-D1*tau))/(gamma*gamma*(1.0-g*np.exp(-D1*tau)))\
        *(kappa-gamma*rhoxv*i*u-D1)
    return D
    
def A_H1HW(u,tau,P0T,lambd,eta,kappa,gamma,vbar,v0,rhoxv,rhoxr):
    i  = np.complex(0.0,1.0)
    D1 = np.sqrt(np.power(kappa-gamma*rhoxv*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = (kappa-gamma*rhoxv*i*u-D1)/(kappa-gamma*rhoxv*i*u+D1)
    
    # Function theta(t)

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2.0*dt)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))  

    # Integration within the function I_1

    N  = 500
    z  = np.linspace(0,tau-1e-10,N)
    f1 = (1.0-np.exp(-lambd*z))*theta(tau-z)
    value1 = integrate.trapz(f1,z)
    
    # Note that I_1_adj also allows time-dependent theta 
    # Therefore it is not exactly the same as in the book

    I_1_adj = (i*u-1.0) * value1
    I_2     = tau/(gamma**2.0) *(kappa-gamma*rhoxv*i*u-D1) - 2.0/(gamma**2.0)*np.log((1.0-g*np.exp(-D1*tau))/(1.0-g))
    I_3     = 1.0/(2.0*np.power(lambd,3.0))* np.power(i+u,2.0)*(3.0+np.exp(-2.0*lambd*tau)-4.0*np.exp(-lambd*tau)-2.0*lambd*tau)
    
    meanSqrtV = meanSqrtV_3(kappa,v0,vbar,gamma)
    f2        = meanSqrtV(tau-z)*(1.0-np.exp(-lambd*z))
    value2    = integrate.trapz(f2,z)
    I_4       = -1.0/lambd * (i*u+u**2.0)*value2
    
    return I_1_adj + kappa*vbar*I_2 + 0.5*eta**2.0*I_3+eta*rhoxr*I_4

def ChFH1HWModel(P0T,lambd,eta,tau,kappa,gamma,vbar,v0,rhoxv, rhoxr):
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2.0*dt)
    r0 =f0T(0.00001)
    C = lambda u: C_H1HW(u,tau,lambd)
    D = lambda u: D_H1HW(u,tau,kappa,gamma,rhoxv)
    A = lambda u: A_H1HW(u,tau,P0T,lambd,eta,kappa,gamma,vbar,v0,rhoxv,rhoxr)
    cf = lambda u: np.exp(A(u) + C(u)*r0 + D(u)*v0 )
    return cf

# Calibration of the SZHW model 

def calibrationSZHW(CP,kappa,Rxr,eta,lambd,K,marketPrice,S0,T,P0T):
    K = np.array(K)
    marketPrice = np.array(marketPrice)
    # x = [gamma,sigmaBar,Rrsigma,Rxsigma,sigma0]
    f_obj = lambda x: TargetValSZHW(CP,kappa,x[0],x[1],Rxr,x[2],x[3],x[4],eta,lambd,K,marketPrice,S0,T,P0T)
    
    # Random initial guess

    #[gamma,sigmaBar,Rrsigma,Rxsigma,sigma0
    initial = np.array([0.1, 0.1, 0.0,-0.5,0.1])
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

def calibrationSZHW_Global(CP,kappa,Rxr,eta,lambd,K,marketPrice,S0,T,P0T):
    K = np.array(K)
    marketPrice = np.array(marketPrice)
    # x = [gamma,sigmaBar,Rrsigma,Rxsigma,sigma0]
    f_obj = lambda x: TargetValSZHW(CP,kappa,x[0],x[1],Rxr,x[2],x[3],x[4],eta,lambd,K,marketPrice,S0,T,P0T)
    
    # Random initial guess

    #[gamma,sigmaBar,Rrsigma,Rxsigma,sigma0
    initial = np.array([0.1, 0.1,0.0,-0.5,0.3])
    
    # The bounds

    xmin = [0.1, 0.01,-0.85, -0.85, 0.01]
    xmax = [0.8,  0.4, 0.85, -0.0, 0.8]    
    
    # Rewrite the bounds as required by L-BFGS-B
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

# Global calibration of the Heston Hull-White model 

def calibrationH1HW_Global(CP,kappa,Rxr,eta,lambd,K,marketPrice,S0,T,P0T):
    K = np.array(K)
    marketPrice = np.array(marketPrice)
    # x = [gamma,vBar,Rxv,v0]
    f_obj = lambda x: TargetValH1HW(CP,kappa,x[0],x[1],Rxr,x[2],x[3],eta,lambd,K,marketPrice,S0,T,P0T)
       
    # Random initial guess

    #[gamma,vBar,Rxv,v0]
    initial = np.array([1.0, 0.05,-0.7, 0.05])
    
    # The bounds

    xmin = [0.1, 0.001,-0.99, 0.001]
    xmax = [0.8,  0.4,  -0.3, 0.4]    
    
    # Rewrite the bounds as required by L-BFGS-B

    bounds = [(low, high) for low, high in zip(xmin, xmax)]

    # Use L-BFGS-B method because the problem is smooth and bounded

    minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds,options={'niter':1})

    # Global search

    pars = optimize.basinhopping(f_obj, initial,niter=1, minimizer_kwargs=minimizer_kwargs)
    
    print(pars)
    
    # Use global parameters in the local search

    gamma_est = pars.x[0]
    vBar_est = pars.x[1]
    Rxv_est = pars.x[2]
    v0_est = pars.x[3]
    initial = [gamma_est,vBar_est,Rxv_est,v0_est]    
    pars  = minimize(f_obj,initial,method='nelder-mead', options = \
                     {'xtol': 1e-05, 'disp': False,'maxiter':200})
    
    gamma_est = pars.x[0]
    vBar_est = pars.x[1]
    Rxv_est = pars.x[2]
    v0_est = pars.x[3]
    parmCalibr =  {"gamma":gamma_est,"vBar":vBar_est,"Rxv":Rxv_est,\
                   "v0":v0_est,'ErrorFinal':pars.fun}
    return parmCalibr

# Calibration of the Heston Hull-White model 

def calibrationH1HW(CP,kappa,Rxr,eta,lambd,K,marketPrice,S0,T,P0T):
    K = np.array(K)
    marketPrice = np.array(marketPrice)
    # x = [gamma,vBar,Rxv,v0]
    f_obj = lambda x: TargetValH1HW(CP,kappa,x[0],x[1],Rxr,x[2],x[3],eta,lambd,K,marketPrice,S0,T,P0T)
       
    # Random initial guess

    #[gamma,vBar,Rxv,v0]
    initial = np.array([1.0, 0.05,-0.7, 0.05])
    pars  = minimize(f_obj,initial,method='nelder-mead', options = \
                     {'xtol': 1e-05, 'disp': False,'maxiter':100})
    print(pars)
    
    gamma_est = pars.x[0]
    vBar_est = pars.x[1]
    Rxv_est = pars.x[2]
    v0_est = pars.x[3]
    parmCalibr =  {"gamma":gamma_est,"vBar":vBar_est,"Rxv":Rxv_est,\
                   "v0":v0_est,'ErrorFinal':pars.fun}
    return parmCalibr

def TargetValSZHW(CP,kappa,gamma,sigmaBar,Rxr,Rrsigma,Rxsigma,sigma0,eta,lambd,K,marketPrice,S0,T,P0T):
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    # Setting additional constraints

    if gamma>1.0 or sigmaBar>1.0 or np.abs(Rxsigma)>1.0:
        return 10e10
    
    # Settings for the COS method

    N = 500
    L = 15  
    cf = lambda u: ChFSZHW(u,P0T,sigma0,T,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmaBar)
    valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
    
    # Error is defined as the difference between the market and the model

    errorVector = valCOS - marketPrice
    
    # Target value is a norm of the error vector

    value       = np.linalg.norm(errorVector)   
    print("Total Error = {0}".format(value))
    return value

def TargetValH1HW(CP,kappa,gamma,vBar,Rxr,Rxv,v0,eta,lambd,K,marketPrice,S0,T,P0T):
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    # Settings for the COS method

    N = 2000
    L = 15  
    cf = ChFH1HWModel(P0T,lambd,eta,T,kappa,gamma,vBar,v0, Rxv, Rxr)
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

    lambd = 0.01
    eta   = 0.01
    S0    = 1145.88       
   
    # Fixed SZHW / H1-HW model parameters

    kappa =  0.5
    Rxr   =  0.3
      
    # We define a ZCB curve (obtained from the market)
    # This curve is based on the estimation to real market data

    P0T = lambda T: np.exp(0.0012*T+0.0007) 
       
    # Settings for the COS method

    N = 2000
    L = 15  
    
    ################## Here we define market call option prices #################
    
    T = 0.5
    referenceIV  = np.array([57.61, 22.95, 15.9])/100.0
    
    #T = 1.0
    #referenceIV = np.array([48.53, 24.49, 19.23])/100.0
        
    #T = 10.0
    #referenceIV = np.array([36.76, 29.18, 27.66])/100.0
    
    # Strike range

    frwd = S0/P0T(T)
    K = [0.4*frwd, frwd, 1.2*frwd]
    K = np.array(K).reshape([len(K),1])
    
    referenceIV = np.array(referenceIV).reshape([len(referenceIV),1])
    referencePrice = P0T(T)* BS_Call_Put_Option_Price(CP,S0 / P0T(T),K,referenceIV,T,0.0)
    
    
    plt.figure(1)
    plt.title('Calibration of the SZHW model')
    plt.plot(K,referencePrice)
    plt.xlabel('strike, K')
    plt.ylabel('reference option price')
    plt.grid()
    
    # Calibrate the SZHW model and show the output

    calibratedParms = calibrationSZHW_Global(CP,kappa,Rxr,eta,lambd,K,referencePrice,S0,T,P0T)

    # Plot calibrated SZHW prices          

    gamma    = calibratedParms.get('gamma')
    sigmaBar = calibratedParms.get('sigmaBar')
    Rrsigma  = calibratedParms.get('Rrsigma')
    Rxsigma  = calibratedParms.get('Rxsigma')
    sigma0   = calibratedParms.get('sigma0')   
    errorSZHW =calibratedParms.get('ErrorFinal')   
    cf = lambda u: ChFSZHW(u,P0T,sigma0,T,lambd,gamma,Rxsigma,Rrsigma,Rxr,eta,kappa,sigmaBar)
    valCOS_SZHW = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))
    plt.plot(K,valCOS_SZHW,'--r')
    plt.legend(['Input','Calibration output'])
    
    print("Optimal parameters for SZHW are: gamma = {0:.3f}, sigmaBar = {1:.3f}, Rrsigma \
          = {2:.3f}, Rxsigma = {3:.3f}, sigma0 = {4:.3f}".format(gamma,\
          sigmaBar,Rrsigma,Rxsigma,sigma0))
        
    # Calibrate the H1HW model and show the output

    plt.figure(2)
    plt.title('Calibration of the H1HW model')
    plt.plot(K,referencePrice)
    plt.xlabel('strike, K')
    plt.ylabel('reference option price')
    plt.grid()
    
    calibratedParms =  calibrationH1HW_Global(CP,kappa,Rxr,eta,lambd,K,referencePrice,S0,T,P0T)
    
    gamma = calibratedParms.get('gamma')
    vBar  = calibratedParms.get('vBar')
    Rxv   = calibratedParms.get('Rxv')
    v0    = calibratedParms.get('v0')
    errorH1HW = calibratedParms.get('ErrorFinal')   
    
    cf2 = ChFH1HWModel(P0T,lambd,eta,T,kappa,gamma,vBar,v0,Rxv, Rxr)
    valCOS_H1HW = CallPutOptionPriceCOSMthd_StochIR(cf2, CP, S0, T, K, N, L,P0T(T))
    
    plt.plot(K,valCOS_H1HW,'--r')
    plt.legend(['Input','Calibration output'])
    
    
    print("Optimal parameters for H1-HW are: gamma = {0:.3f}, vBar = {1:.3f}, Rxv = {2:.3f}, v0 = {3:.3f}".format(gamma,vBar,Rxv,v0))
    print("=======================================================================")
    print('Final error for SZHW ={0} and for H1HW={1}'.format(errorSZHW,errorH1HW))
    
    # Plot implied volatilities for both models    

    IVH1HW =np.zeros([len(K),1])
    IVSZHW =np.zeros([len(K),1])
    IVMarket =np.zeros([len(K),1])
    for (idx,k) in enumerate(K):
        IVMarket[idx] = ImpliedVolatilityBlack76(CP,referencePrice[idx]/P0T(T),k,T,S0/P0T(T))*100.0
        IVH1HW[idx] = ImpliedVolatilityBlack76(CP,valCOS_H1HW[idx]/P0T(T),k,T,S0/P0T(T))*100.0
        IVSZHW[idx] = ImpliedVolatilityBlack76(CP,valCOS_SZHW[idx]/P0T(T),k,T,S0/P0T(T))*100.0
    
    plt.figure(3)
    plt.plot(K,IVMarket)
    plt.plot(K,IVH1HW,'--r')
    plt.plot(K,IVSZHW)
    plt.grid()
    plt.legend(['market','H1HW','SZHW'])
    
    print('Market IV')
    print(IVMarket)
    print('H1HW IV')
    print(IVH1HW)
    print('SZHW IV')
    print(IVSZHW)
    
mainCalculation()
