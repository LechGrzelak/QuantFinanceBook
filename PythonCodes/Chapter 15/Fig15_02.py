#%%
"""
Created on Feb 25 2019
The calibration of the FX-HHW model. The calibration is performed using a global and local search algoritm.
In the simulation we calibrate the model at one expiry at the time. The framework can be 
easily extended to the calibration to the whole volatility surface.
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
    # To determine initial volatility we interpolate define a grid for sigma
    # and interpolate on the inverse
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
       value[k]=(value1 + value2*i)
    
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

def C_H1HW_FX(u,tau,kappa,gamma,rhoxv):
    i = np.complex(0.0,1.0)
    
    D1 = np.sqrt(np.power(kappa-gamma*rhoxv*i*u,2.0)+(u*u+i*u)*gamma*gamma)
    g  = (kappa-gamma*rhoxv*i*u-D1)/(kappa-gamma*rhoxv*i*u+D1)
    C  = (1.0-np.exp(-D1*tau))/(gamma*gamma*(1.0-g*np.exp(-D1*tau)))\
        *(kappa-gamma*rhoxv*i*u-D1)
    return C

def ChFH1HW_FX(u,tau,gamma,Rxv,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdd,etad,lambdf,etaf,kappa,vBar,v0):
    i  = np.complex(0.0,1.0)
    C = lambda u,tau: C_H1HW_FX(u,tau,kappa,gamma,Rxv)
    Bd = lambda t,T: 1.0/lambdd*(np.exp(-lambdd*(T-t))-1.0)
    Bf = lambda t,T: 1.0/lambdf*(np.exp(-lambdf*(T-t))-1.0)
    G = meanSqrtV_3(kappa,v0,vBar,gamma)
    
    zeta = lambda t: (Rxrd*etad*Bd(t,tau) - Rxrf*etaf*Bf(t,tau))*G(t) + \
                    Rrdrf*etad*etaf*Bd(t,tau)*Bf(t,tau) - 0.5*(etad**2.0*Bd(t,tau)**2.0+etaf**2.0*Bf(t,tau)**2.0)
    
    # Integration within the function A(u,tau)

    int1=np.zeros([len(u),1],dtype=np.complex_)   
    N = 100
    z=np.linspace(0.0+1e-10,tau-1e-10,N)
    
    temp1 =lambda z1: kappa*vBar + Rvrd*gamma*etad*G(tau-z1)*Bd(tau-z1,tau)
    temp2 =lambda z1, u: -Rvrd*gamma*etad*G(tau-z1)*Bd(tau-z1,tau)*i*u
    temp3 =lambda z1, u:  Rvrf*gamma*etaf*G(tau-z1)*Bf(tau-z1,tau)*i*u
    f = lambda z1,u: (temp1(z1)+temp2(z1,u)+temp3(z1,u))*C(u,z1)
    
    
    value1 =integrate.trapz(np.real(f(z,u)),z).reshape(u.size,1)
    value2 =integrate.trapz(np.imag(f(z,u)),z).reshape(u.size,1)
    int1=(value1 + value2*i)
    
    """
    for k in range(0,len(u)):
        temp1 = kappa*vBar + Rvrd*gamma*etad*G(tau-z)*Bd(tau-z,tau)
        temp2 = -Rvrd*gamma*etad*G(tau-z)*Bd(tau-z,tau)*i*u[k]
        temp3 = Rvrf*gamma*etaf*G(tau-z)*Bf(tau-z,tau)*i*u[k]
        f = (temp1+temp2+temp3)*C(u[k],z)
        value1 =integrate.trapz(np.real(f),z)
        value2 =integrate.trapz(np.imag(f),z)
        int1[k]=(value1 + value2*i)
    """
    int2 = (u**2.0 + i*u)*integrate.trapz(zeta(tau-z),z)
    
    A = int1 + int2
    
    cf = np.exp(A + v0*C(u,tau))
    return cf

# Global calibration of the Heston Hull-White model 

def calibrationH1HW_FX_Global(CP,marketPrice,P0Td,T,K,frwdFX,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdd,etad,lambdf,etaf,kappa):
    K = np.array(K)
    marketPrice = np.array(marketPrice)
    # x = [gamma,vBar,Rxv,v0]
    f_obj = lambda x: TargetValH1HW_FX(CP,marketPrice,P0Td,T,K,frwdFX,x[0],x[1],x[2],x[3],Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdd,etad,lambdf,etaf,kappa)
       
    # Random initial guess

    #[gamma,vBar,Rxv,v0]
    initial = np.array([1.0, 0.005,-0.7, 0.004])
    
    # The bounds

    xmin = [0.1, 0.001,-0.99, 0.0001]
    xmax = [0.8,  0.4,  -0.3, 0.4]    
    
    # Rewrite the bounds in the way required by L-BFGS-B

    bounds = [(low, high) for low, high in zip(xmin, xmax)]

    # Use L-BFGS-B method, because the problem is smooth and bounded

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
                     {'xtol': 1e-05, 'disp': False,'maxiter':100})
    
    gamma_est = pars.x[0]
    vBar_est = pars.x[1]
    Rxv_est = pars.x[2]
    v0_est = pars.x[3]
    parmCalibr =  {"gamma":gamma_est,"vBar":vBar_est,"Rxv":Rxv_est,\
                   "v0":v0_est,'ErrorFinal':pars.fun}
    return parmCalibr

def TargetValH1HW_FX(CP,marketPrice,P0Td,T,K,frwdFX,gamma,vbar,Rxv,v0,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdd,etad,lambdf,etaf,kappa):
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    # Settings for the COS method

    N = 2000
    L = 15  

    #cf = ChFH1HWModel(P0T,lambd,eta,T,kappa,gamma,vBar,v0, Rxv, Rxr)
    #valCOS = CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, T, K, N, L,P0T(T))

    cf = lambda u: ChFH1HW_FX(u,T,gamma,Rxv,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdd,etad,lambdf,etaf,kappa,vbar,v0)    
    
    valCOS = P0Td(T)*CallPutOptionPriceCOSMthd_StochIR(cf, CP, frwdFX, T, K, N, L,1.0)
    
    # Error is defined as the difference between the market and the model

    errorVector = valCOS - marketPrice
    
    # Target value is a norm of the error vector

    value       = np.linalg.norm(errorVector)   
    print("Total Error = {0}".format(value))
    return value

def GenerateStrikes(frwd,Ti):
    c_n = np.array([-1.5, -1.0, -0.5,0.0, 0.5, 1.0, 1.5])
    return frwd * np.exp(0.1 * c_n * np.sqrt(Ti))


def mainCalculation():
    CP  = OptionType.CALL  

    # Specify the maturity for the calibration

    index = 1
    
    T_market  = [0.5,1.0,5.0,10.0,20.0,30.0]
    IV_Market = np.array([[11.41 , 10.49 , 9.66 , 9.02 , 8.72 , 8.66 , 8.68 ],
                 [12.23 , 10.98 , 9.82 , 8.95 , 8.59 , 8.59 , 8.65],
                 [13.44 , 11.84 , 10.38 , 9.27 , 8.76 , 8.71 , 8.83],
                 [16.43 , 14.79 , 13.34 , 12.18 , 11.43 , 11.07 , 10.99],
                 [22.96 , 21.19 , 19.68 , 18.44 , 17.50 , 16.84 , 16.46],
                 [25.09 , 23.48 , 22.17 , 21.13 , 20.35 , 19.81 , 19.48]])
    T = T_market[index]
    referenceIV = IV_Market[index,:]/100.0
    
    # Settings for the COS method

    N = 500
    L = 8  
    
    # ZCBs from the domestic and foreign markets

    P0Td = lambda t: np.exp(-0.02*t)
    P0Tf = lambda t: np.exp(-0.05*t)
    
    y0      = 1.35
    frwdFX  = y0*P0Tf(T)/P0Td(T)
    
    # Fixed mean-reversion parameter

    kappa =0.5
    
    # HW model parameters

    lambdd  = 0.001
    lambdf  = 0.05
    etad    = 0.007
    etaf    = 0.012
    
    # Correlations

    Rxrd  = -0.15
    Rxrf  = -0.15
    Rvrd  = 0.3
    Rvrf  = 0.3
    Rrdrf = 0.25
    
    # Strikes and option prices from market implied volatilities

    K = GenerateStrikes(frwdFX,T)
    K = np.array(K).reshape([len(K),1])
    referenceIV = np.array(referenceIV).reshape([len(referenceIV),1])
    referencePrice = P0Td(T)* BS_Call_Put_Option_Price(CP,frwdFX,K,referenceIV,T,0.0)
       
    # Calibrate the H1HW model and show the output

    plt.figure(1)
    plt.title('Calibration of the H1HW model')
    plt.plot(K,referencePrice)
    plt.xlabel('strike, K')
    plt.ylabel('reference option price')
    plt.grid()
    
    # Calibrate the hybrid model

    calibratedParms =  calibrationH1HW_FX_Global(CP,referencePrice,P0Td,T,K,frwdFX,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdd,etad,lambdf,etaf,kappa)
    
    gamma = calibratedParms.get('gamma')
    vBar  = calibratedParms.get('vBar')
    Rxv   = calibratedParms.get('Rxv')
    v0    = calibratedParms.get('v0')
    errorH1HW = calibratedParms.get('ErrorFinal')   
    
    cf = lambda u: ChFH1HW_FX(u,T,gamma,Rxv,Rxrd,Rxrf,Rrdrf,Rvrd,Rvrf,lambdd,etad,lambdf,etaf,kappa,vBar,v0)    
    valCOS_H1HW = P0Td(T)*CallPutOptionPriceCOSMthd_StochIR(cf, CP, frwdFX, T, K, N, L,1.0)
    
    plt.plot(K,valCOS_H1HW,'--r')
    plt.legend(['Input','Calibration output'])
    
    print("Optimal parameters for HHW-FX are: gamma = {0:.3f}, vBar = {1:.3f}, Rxv = {2:.3f}, v0 = {3:.3f}".format(gamma,vBar,Rxv,v0))
    print("=======================================================================")
    print('Final error for H1HW={0}'.format(errorH1HW))
    
    # Plot the implied volatilities for both models    

    IVH1HW =np.zeros([len(K),1])
    IVMarket =np.zeros([len(K),1])
    for (idx,k) in enumerate(K):
        priceCOS = valCOS_H1HW[idx]/P0Td(T) 
        IVH1HW[idx] = ImpliedVolatilityBlack76(CP,priceCOS,k,T,frwdFX)*100.0
        priceMarket = referencePrice[idx]/P0Td(T)
        IVMarket[idx] = ImpliedVolatilityBlack76(CP,priceMarket,k,T,frwdFX)*100.0
    
    plt.figure(2)
    plt.plot(K,IVMarket)
    plt.plot(K,IVH1HW)
    plt.grid()
    plt.xlabel('strike')
    plt.ylabel('Implied volatility')
    plt.legend(['Market IV','HHW-FX-FX IV'])
    plt.title('Implied volatility for T={0}'.format(T))
    
mainCalculation()
