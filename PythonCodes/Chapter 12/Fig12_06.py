#%%
"""
Created on Feb 19 2019
Exposures for an IR swap, under the Hull-White model - case of netting
@author: Lech A. Grzelak
"""
import numpy as np
import enum 
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
import scipy.optimize as optimize

# This class defines puts and calls

class OptionTypeSwap(enum.Enum):
    RECEIVER = 1.0
    PAYER = -1.0

def GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambd, eta):    

    # Time step

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

        # Making sure that samples from a normal have mean 0 and variance 1

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
    n = np.size(rT1) 
        
    if T1<T2:
        B_r = HW_B(lambd,eta,T1,T2)
        A_r = HW_A(lambd,eta,P0T,T1,T2)
        return np.exp(A_r + B_r *rT1)
    else:
        return np.ones([n])


def HWMean_r(P0T,lambd,eta,T):

    # Time step 

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

    # Time step 

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)

    # Initial interest rate is forward rate at time t->0

    r0 = f0T(0.00001)
    return r0

def HW_Mu_FrwdMeasure(P0T,lambd,eta,T):

    # Time step 

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

def HW_SwapPrice(CP,notional,K,t,Ti,Tm,n,r_t,P0T,lambd,eta):

    # CP -- Payer of receiver
    # n --  Notional
    # K --  Strike
    # t --  Today's date
    # Ti -- Beginning of the swap
    # Tm -- End of Swap
    # n --  Number of dates payments between Ti and Tm
    # r_t - Interest rate at time t

    if n == 1:
        ti_grid =np.array([Ti,Tm])
    else:
        ti_grid = np.linspace(Ti,Tm,n)
    tau = ti_grid[1]- ti_grid[0]
    
    # Overwrite Ti if t>Ti

    prevTi = ti_grid[np.where(ti_grid<t)]
    if np.size(prevTi) > 0: #prevTi != []:
        Ti = prevTi[-1]
    
    # We need to handle the case when some payments are already done

    ti_grid = ti_grid[np.where(ti_grid>t)]          

    temp= np.zeros(np.size(r_t));
    
    P_t_TiLambda = lambda Ti : HW_ZCB(lambd,eta,P0T,t,Ti,r_t)
    
    for (idx,ti) in enumerate(ti_grid):
        if ti>Ti:
            temp = temp + tau * P_t_TiLambda(ti)
            
    P_t_Ti = P_t_TiLambda(Ti)
    P_t_Tm = P_t_TiLambda(Tm)
    
    if CP==OptionTypeSwap.PAYER:
        swap = (P_t_Ti - P_t_Tm) - K * temp
    elif CP==OptionTypeSwap.RECEIVER:
        swap = K * temp- (P_t_Ti - P_t_Tm)
    
    return swap*notional

def mainCalculation():
    NoOfPaths = 2000
    NoOfSteps = 1000
    CP = OptionTypeSwap.PAYER
    lambd     = 0.5
    eta       = 0.03
    notional  = 10000.0 
    notional2 = 10000.0
    alpha     = 0.99
    alpha2     = 0.95
    
    # We define a ZCB curve (obtained from the market)

    P0T = lambda T: np.exp(-0.01*T)#np.exp(-0.03*T*T-0.1*T)
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
    
    
    # Here we simulate the exposure profiles for a swap, using the HW model    
    # Swap settings

    K = 0.01  # Strike
    Ti = 1.0  # Beginning of the swap
    Tm = 10.0 # End date of the swap 
    n = 10    # Number of payments between Ti and Tm
    
    paths= GeneratePathsHWEuler(NoOfPaths,NoOfSteps,Tm+1.0 ,P0T, lambd, eta)
    r = paths["R"]
    timeGrid = paths["time"]
    dt = timeGrid[1]-timeGrid[0]
    
    # Here we compare the price of an option on a ZCB from Monte Carlo and analytic expression    

    M_t = np.zeros([NoOfPaths,NoOfSteps])
            
    for i in range(0,NoOfPaths):
        M_t[i,:] = np.exp(np.cumsum(r[i,0:-1])*dt)
        
    # Portfolio without netting    

    Value= np.zeros([NoOfPaths,NoOfSteps+1])
    E  = np.zeros([NoOfPaths,NoOfSteps+1])
    EE = np.zeros([NoOfSteps+1])
    PFE = np.zeros([NoOfSteps+1])
    PFE2 = np.zeros([NoOfSteps+1])
    for (idx, ti) in enumerate(timeGrid[0:-2]):
        V = HW_SwapPrice(CP,notional,K,timeGrid[idx],Ti,Tm,n,r[:,idx],P0T,lambd,eta)
        Value[:,idx] = V / np.squeeze(M_t[:,idx])
        E[:,idx] = np.maximum(V,0.0)
        EE[idx] = np.mean(E[:,idx]/M_t[:,idx])
        PFE[idx] = np.quantile(E[:,idx],alpha)
        PFE2[idx] = np.quantile(E[:,idx],alpha2)
    
    # Portfolio with netting    

    ValuePort = np.zeros([NoOfPaths,NoOfSteps+1])
    EPort  = np.zeros([NoOfPaths,NoOfSteps+1])
    EEPort = np.zeros([NoOfSteps+1])
    PFEPort = np.zeros([NoOfSteps+1])
    for (idx, ti) in enumerate(timeGrid[0:-2]):
        Swap1 = HW_SwapPrice(CP,notional,K,timeGrid[idx],Ti,Tm,n,r[:,idx],P0T,lambd,eta)
        Swap2 = HW_SwapPrice(CP,notional2,0.0,timeGrid[idx],Tm-2.0*(Tm-Ti)/n,Tm,1,r[:,idx],P0T,lambd,eta)
        
        VPort = Swap1 - Swap2
        ValuePort[:,idx] = VPort / np.squeeze(M_t[:,idx])
        EPort[:,idx] = np.maximum(VPort,0.0)
        EEPort[idx] = np.mean(EPort[:,idx]/M_t[:,idx])
        PFEPort[idx] = np.quantile(EPort[:,idx],alpha)
    
    plt.figure(2)
    plt.plot(timeGrid,Value[0:100,:].transpose(),'b')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('exposure, Value(t)')
    plt.title('Value of a swap')

    plt.figure(3)
    plt.plot(timeGrid,E[0:100,:].transpose(),'r')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('exposure, E(t)')
    plt.title('Positive Exposure E(t)')
    
    plt.figure(4)
    plt.plot(timeGrid,EE,'r')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('exposure, EE(t)')
    plt.title('Discounted Expected (positive) exposure, EE')
    plt.legend(['EE','PFE'])
    
    plt.figure(5)
    plt.plot(timeGrid,EE,'r')
    plt.plot(timeGrid,PFE,'k')
    plt.plot(timeGrid,PFE2,'--b')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel(['EE, PEE(t)'])
    plt.title('Discounted Expected (positive) exposure, EE')
    
    plt.figure(6)
    plt.plot(timeGrid,EEPort,'r')
    plt.plot(timeGrid,PFEPort,'k')
    plt.grid()
    plt.title('Portfolio with two swaps')
    plt.legend(['EE-port','PFE-port'])
    
    plt.figure(7)
    plt.plot(timeGrid,EE,'r')
    plt.plot(timeGrid,EEPort,'--r')
    plt.grid()
    plt.title('Comparison of EEs ')
    plt.legend(['EE, swap','EE, portfolio'])
    
    plt.figure(8)
    plt.plot(timeGrid,PFE,'k')
    plt.plot(timeGrid,PFEPort,'--k')
    plt.grid()
    plt.title('Comparison of PFEs ')
    plt.legend(['PFE, swap','PFE, portfolio'])
    
mainCalculation()
