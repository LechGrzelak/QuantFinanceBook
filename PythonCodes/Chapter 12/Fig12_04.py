#%%
"""
Created on Jan 25 2019
Pricing of a caplet/floorlet with the Hull-White model
@author: Lech A. Grzelak
"""
import numpy as np
import enum 
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
import scipy.optimize as optimize

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambd, eta):    

    # Time step needed for differentiation

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    
    # Initial interest rate is forward rate at time t->0

    r0 = f0T(0.00001)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))      
    
    #theta = lambda t: 0.1 +t -t
    #print("changed theta")
    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    R = np.zeros([NoOfPaths, NoOfSteps+1])
    R[:,0]=r0
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        R[:,i+1] = R[:,i] + lambd*(theta(time[i]) - R[:,i]) * dt + eta* (W[:,i+1]-W[:,i])
        time[i+1] = time[i] +dt
        
    # Output

    paths = {"time":time,"R":R}
    return paths

def HW_theta(lambd,eta,P0T):
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    theta = lambda t: 1.0/lambd * (f0T(t+dt)-f0T(t-dt))/(2.0*dt) + f0T(t) + eta*eta/(2.0*lambd*lambd)*(1.0-np.exp(-2.0*lambd*t))
    #print("CHANGED THETA")
    return theta#lambda t: 0.1+t-t
    
def HW_A(lambd,eta,P0T,T1,T2):
    tau = T2-T1
    zGrid = np.linspace(0.0,tau,250)
    B_r = lambda tau: 1.0/lambd * (np.exp(-lambd *tau)-1.0)
    theta = HW_theta(lambd,eta,P0T)    
    temp1 = lambd * integrate.trapz(theta(T2-zGrid)*B_r(zGrid),zGrid)
    
    temp2 = eta*eta/(4.0*np.power(lambd,3.0)) * (np.exp(-2.0*lambd*tau)*(4*np.exp(lambd*tau)-1.0) -3.0) + eta*eta*tau/(2.0*lambd*lambd)
    
    return temp1 + temp2

def HW_B(lambd,eta,T1,T2):
    return 1.0/lambd *(np.exp(-lambd*(T2-T1))-1.0)

def HW_ZCB(lambd,eta,P0T,T1,T2,rT1):
    B_r = HW_B(lambd,eta,T1,T2)
    A_r = HW_A(lambd,eta,P0T,T1,T2)
    return np.exp(A_r + B_r *rT1)

def HWMean_r(P0T,lambd,eta,T):

    # Time step needed for differentiation

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2.0*dt)

    # Initial interest rate is forward rate at time t->0

    r0 = f0T(0.00001)
    theta = HW_theta(lambd,eta,P0T)
    zGrid = np.linspace(0.0,T,2500)
    temp =lambda z: theta(z) * np.exp(-lambd*(T-z))
    r_mean = r0*np.exp(-lambd*T) + lambd * integrate.trapz(temp(zGrid),zGrid)
    return r_mean

def HW_r_0(P0T,lambd,eta):

    # Time step needed for differentiation

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)

    # Initial interest rate is forward rate at time t->0

    r0 = f0T(0.00001)
    return r0

def HW_Mu_FrwdMeasure(P0T,lambd,eta,T):

    # Time step needed for differentiation

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)

    # Initial interest rate is forward rate at time t->0

    r0 = f0T(0.00001)
    theta = HW_theta(lambd,eta,P0T)
    zGrid = np.linspace(0.0,T,500)
    
    theta_hat =lambda t,T:  theta(t) + eta*eta / lambd *1.0/lambd * (np.exp(-lambd*(T-t))-1.0)
    
    temp =lambda z: theta_hat(z,T) * np.exp(-lambd*(T-z))
    
    r_mean = r0*np.exp(-lambd*T) + lambd * integrate.trapz(temp(zGrid),zGrid)
    
    return r_mean

def HWVar_r(lambd,eta,T):
    return eta*eta/(2.0*lambd) *( 1.0-np.exp(-2.0*lambd *T))

def HWDensity(P0T,lambd,eta,T):
    r_mean = HWMean_r(P0T,lambd,eta,T)
    r_var = HWVar_r(lambd,eta,T)
    return lambda x: st.norm.pdf(x,r_mean,np.sqrt(r_var))

def HW_CapletFloorletPrice(CP,N,K,lambd,eta,P0T,T1,T2):
    if CP == OptionType.CALL:
        N_new = N * (1.0+(T2-T1)*K)
        K_new = 1.0 + (T2-T1)*K
        caplet = N_new*HW_ZCB_CallPutPrice(OptionType.PUT,1.0/K_new,lambd,eta,P0T,T1,T2)
        value= caplet
    elif CP==OptionType.PUT:
        N_new = N * (1.0+ (T2-T1)*K)
        K_new = 1.0 + (T2-T1)*K
        floorlet = N_new*HW_ZCB_CallPutPrice(OptionType.CALL,1.0/K_new,lambd,eta,P0T,T1,T2)
        value = floorlet
    return value
    
def HW_ZCB_CallPutPrice(CP,K,lambd,eta,P0T,T1,T2):
    B_r = HW_B(lambd,eta,T1,T2)
    A_r = HW_A(lambd,eta,P0T,T1,T2)
    
    mu_r = HW_Mu_FrwdMeasure(P0T,lambd,eta,T1)
    v_r =  np.sqrt(HWVar_r(lambd,eta,T1))
    
    K_hat = K * np.exp(-A_r)
    
    a = (np.log(K_hat) - B_r*mu_r)/(B_r*v_r)
    
    d1 = a - B_r*v_r
    d2 = d1 +  B_r*v_r
    
    term1 = np.exp(0.5* B_r*B_r*v_r*v_r + B_r*mu_r)*st.norm.cdf(d1) - K_hat * st.norm.cdf(d2)    
    value =P0T(T1) * np.exp(A_r) * term1 
    
    if CP == OptionType.CALL:
        return value
    elif CP==OptionType.PUT:
        return value - P0T(T2) + K*P0T(T1)

# Black-Scholes call option price

def BS_Call_Put_Option_Price(CP,S_0,K,sigma,tau,r):
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * tau) / (sigma * np.sqrt(tau))
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

def mainCalculation():
    CP= OptionType.CALL
    NoOfPaths = 20000
    NoOfSteps = 1000
        
    lambd     = 0.02
    eta       = 0.02
    
    # We define a ZCB curve (obtained from the market)

    P0T = lambda T: np.exp(-0.1*T)#np.exp(-0.03*T*T-0.1*T)
    r0 = HW_r_0(P0T,lambd,eta)
    
    # In this experiment we compare for the ZCB the market and analytic expressions

    N = 25
    T_end = 50
    Tgrid= np.linspace(0,T_end,N)
    
    Exact = np.zeros([N,1])
    Proxy= np.zeros ([N,1])
    for i,Ti in enumerate(Tgrid):
        Proxy[i] = HW_ZCB(lambd,eta,P0T,0.0,Ti,r0)
        Exact[i] = P0T(Ti)
        
    plt.figure(1)
    plt.grid()
    plt.plot(Tgrid,Exact,'-k')
    plt.plot(Tgrid,Proxy,'--r')
    plt.legend(["Analytcal ZCB","Monte Carlo ZCB"])
    plt.title('P(0,T) from Monte Carlo vs. Analytical expression')

    # In this experiment we compare Monte Carlo results for 

    T1 = 4.0
    T2 = 8.0
    
    paths= GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T1 ,P0T, lambd, eta)
    r = paths["R"]
    timeGrid = paths["time"]
    dt = timeGrid[1]-timeGrid[0]
    
    # Here we compare the price of an option on a ZCB from Monte Carlo with the analytic expressions   

    M_t = np.zeros([NoOfPaths,NoOfSteps])
    for i in range(0,NoOfPaths):
        M_t[i,:] = np.exp(np.cumsum(r[i,:-1])*dt)
        
    KVec = np.linspace(0.01,1.7,50)
    Price_MC_V = np.zeros([len(KVec),1])
    Price_Th_V =np.zeros([len(KVec),1])
    P_T1_T2 = HW_ZCB(lambd,eta,P0T,T1,T2,r[:,-1])
    for i,K in enumerate(KVec):
        if CP==OptionType.CALL:
            Price_MC_V[i] =np.mean( 1.0/M_t[:,-1] * np.maximum(P_T1_T2-K,0.0)) 
        elif CP==OptionType.PUT:
            Price_MC_V[i] =np.mean( 1.0/M_t[:,-1] * np.maximum(K-P_T1_T2,0.0)) 
        Price_Th_V[i] =HW_ZCB_CallPutPrice(CP,K,lambd,eta,P0T,T1,T2)#HW_ZCB_CallPrice(K,lambd,eta,P0T,T1,T2)
        
    plt.figure(2)
    plt.grid()
    plt.plot(KVec,Price_MC_V)
    plt.plot(KVec,Price_Th_V,'--r')
    plt.legend(['Monte Carlo','Theoretical'])
    plt.title('Option on ZCB')

    # Effect of the HW model parameters on implied volatilities

    # Define a forward rate between T1 and T2

    frwd = 1.0/(T2-T1) *(P0T(T1)/P0T(T2)-1.0)
    K = np.linspace(frwd/2.0,3.0*frwd,25)
    
    # Effect of eta

    plt.figure(3)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    etaV =[0.01, 0.02, 0.03, 0.04]
    legend = []
    Notional = 1.0
    for etaTemp in etaV:    
       optPrice = HW_CapletFloorletPrice(CP,Notional,K,lambd,etaTemp,P0T,T1,T2)

       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           valCOSFrwd = optPrice[idx]/P0T(T2)/(T2-T1)
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd,K[idx],T2,frwd)
       plt.plot(K,IV*100.0)
       #plt.plot(K,optPrice)
       legend.append('eta={0}'.format(etaTemp))
    plt.legend(legend)        
    
     # Effect of beta

    plt.figure(4)
    plt.grid()
    plt.xlabel('strike, K')
    plt.ylabel('implied volatility')
    lambdaV = [0.01, 0.03, 0.05, 0.09]
    legend = []
    Notional = 1.0
    for lambdTemp in lambdaV:    
       optPrice = HW_CapletFloorletPrice(CP,Notional,K,lambdTemp,eta,P0T,T1,T2)

       # Implied volatilities

       IV =np.zeros([len(K),1])
       for idx in range(0,len(K)):
           valCOSFrwd = optPrice[idx]/P0T(T2)/(T2-T1)
           IV[idx] = ImpliedVolatilityBlack76(CP,valCOSFrwd,K[idx],T2,frwd)
       #plt.plot(K,optPrice)
       plt.plot(K,IV*100.0)
       legend.append('lambda={0}'.format(lambdTemp))
    plt.legend(legend)  

    print('frwd={0}'.format(frwd*P0T(T2)))
mainCalculation()
