#%%
"""
Created on Feb 19 2020
Find optimal weights for the portfolio
@author: Lech A. Grzelak
"""
import numpy as np
import enum 
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize

# This class defines puts and calls
class OptionTypeSwap(enum.Enum):
    RECEIVER = 1.0
    PAYER = -1.0

def GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambd, eta):    
    # time-step needed for differentiation
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    
    # Initial interest rate is a forward rate at time t->0
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
        
    # Outputs
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

def P_t_T(lambd,eta,P0T,T1,T2,rT1):
    n = np.size(rT1) 
        
    if T1<T2:
        B_r = HW_B(lambd,eta,T1,T2)
        A_r = HW_A(lambd,eta,P0T,T1,T2)
        return np.exp(A_r + B_r *rT1)
    else:
        return np.ones([n])

def HW_r_0(P0T,lambd,eta):
    # time-step needed for differentiation
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    return r0

def mainCalculation():
    NoOfPaths = 20000
    NoOfSteps = 1000
    lambd     = 0.05
    eta       = 0.1
    
    # We define a ZCB curve (obtained from the market)
    P0T = lambda T: np.exp(-0.02*T)
    r0 = HW_r_0(P0T,lambd,eta)
    
    # In this experiment we compare ZCB from the Market and Analytical expression
    N = 25
    T_end = 10
    Tgrid= np.linspace(0,T_end,N)
    
    Exact = np.zeros([N,1])
    Proxy= np.zeros ([N,1])
    for i,Ti in enumerate(Tgrid):
        Proxy[i] = P_t_T(lambd,eta,P0T,0.0,Ti,r0)
        Exact[i] = P0T(Ti)
        
    plt.figure(1)
    plt.grid()
    plt.plot(Tgrid,Exact,'-k')
    plt.plot(Tgrid,Proxy,'--r')
    plt.legend(["Analytcal ZCB","Monte Carlo ZCB"])
    plt.title('P(0,T) from Monte Carlo vs. Analytical expression')
    
    # Here we determine optimal weights w_1 and w_2.
    paths =GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T_end,P0T, lambd, eta)
    r_t = paths["R"]
    time_grid = paths["time"]
    
    P_t_1y = np.zeros([NoOfPaths, NoOfSteps+1])
    P_t_10y = np.zeros([NoOfPaths, NoOfSteps+1])
    P_t_20y = np.zeros([NoOfPaths, NoOfSteps+1])

    # Generate all the ZCBs
    for i,t_i in enumerate(time_grid):
        P_t_1y[:,i] = P_t_T(lambd,eta,P0T,t_i,1,r_t[:,i])
        P_t_10y[:,i] = P_t_T(lambd,eta,P0T,t_i,10,r_t[:,i])
        P_t_20y[:,i] = P_t_T(lambd,eta,P0T,t_i,20,r_t[:,i])
    
    # In the second step we integrate each ZCB
    dt = T_end/NoOfSteps
    P_t_1yInt = np.sum(P_t_1y,axis=1)*dt
    P_t_10yInt = np.sum(P_t_10y,axis=1)*dt
    P_t_20yInt = np.sum(P_t_20y,axis=1)*dt
    
    # Final step: determine optimal weight
    var_10y = np.var(P_t_10yInt)
    target = lambda omega1: (np.var(omega1*P_t_1yInt + (1-omega1)*P_t_20yInt) - var_10y)**2.0
    
    omegaGrid = np.linspace(0,1,1000)
    error= np.zeros([len(omegaGrid)])
    for i,omega in enumerate(omegaGrid):
        error[i]= target(omega)
        
    plt.figure(2)
    plt.plot(omegaGrid,error)
    plt.grid()
    plt.xlabel('omega1')
    plt.ylabel('Hedging Variance')
    
    # Optimize and determine optimal omega
    optimalOmega = optimize.minimize(target,0.7)
    print("optimal omega = {0}".format(optimalOmega))
    
mainCalculation()