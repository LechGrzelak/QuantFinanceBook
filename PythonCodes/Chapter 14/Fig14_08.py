#%%
"""
Created on Feb 07 2019
Convexity correction
@author: Lech A. Grzelak
"""
import numpy as np
import enum 
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# This class defines puts and calls

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T,P0T, lambd, eta):    

    # Time step 

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    
    # Initial interest rate is the forward rate at time t->0

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

def HW_ZCB(lambd,eta,P0T,T1,T2,rT1):
    B_r = HW_B(lambd,eta,T1,T2)
    A_r = HW_A(lambd,eta,P0T,T1,T2)
    return np.exp(A_r + B_r *rT1)

def HWMean_r(P0T,lambd,eta,T):

    # time step 
    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2.0*dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = HW_theta(lambd,eta,P0T)
    zGrid = np.linspace(0.0,T,2500)
    temp =lambda z: theta(z) * np.exp(-lambd*(T-z))
    r_mean = r0*np.exp(-lambd*T) + lambd * integrate.trapz(temp(zGrid),zGrid)
    return r_mean

def HW_r_0(P0T,lambd,eta):

    # time step

    dt = 0.0001    
    f0T = lambda t: - (np.log(P0T(t+dt))-np.log(P0T(t-dt)))/(2*dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    return r0

def mainCalculation():
    CP= OptionType.CALL
    NoOfPaths = 20000
    NoOfSteps = 1000
        
    lambd     = 0.02
    eta       = 0.02
    
    # We define a ZCB curve (obtained from the market)

    P0T = lambda T: np.exp(-0.1*T)#np.exp(-0.03*T*T-0.1*T)
    r0 = HW_r_0(P0T,lambd,eta)
    
    # In this experiment we compare for the ZCB market and analytic expressions

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

    # Here we define a Libor rate and measure the convexity effect

    T1 = 4.0
    T2 = 8.0
    
    paths= GeneratePathsHWEuler(NoOfPaths,NoOfSteps,T1 ,P0T, lambd, eta)
    r = paths["R"]
    timeGrid = paths["time"]
    dt = timeGrid[1]-timeGrid[0]
    M_t = np.zeros([NoOfPaths,NoOfSteps])
    for i in range(0,NoOfPaths):
        M_t[i,:] = np.exp(np.cumsum(r[i,:-1])*dt)
    P_T1_T2 = HW_ZCB(lambd,eta,P0T,T1,T2,r[:,-1])
    L_T1_T2 = 1.0/(T2-T1)*(1.0/P_T1_T2-1)
    MC_Result = np.mean(1/M_t[:,-1]*L_T1_T2)
    print('Price of E(L(T1,T1,T2)/M(T1)) = {0}'.format(MC_Result))
    
    L_T0_T1_T2 = 1.0/(T2-T1)*(P0T(T1)/P0T(T2)-1.0)

    # Define the convexity correction

    cc = lambda sigma: P0T(T2)*(L_T0_T1_T2 + (T2-T1)*L_T0_T1_T2**2.0*np.exp(sigma**2*T1))-L_T0_T1_T2
    
    # Take a random sigma and check the effect on the price

    sigma = 0.2
    print('Price of E(L(T1,T1,T2)/M(T1)) = {0} (no cc)'.format(L_T0_T1_T2))
    print('Price of E(L(T1,T1,T2)/M(T1)) = {0} (with cc, sigma={1})'.format(L_T0_T1_T2+cc(sigma),sigma))
    
    # Plot some results

    plt.figure(2)
    sigma_range = np.linspace(0.0,0.6,100)
    plt.plot(sigma_range,cc(sigma_range))
    plt.grid()
    plt.xlabel('sigma')
    plt.ylabel('cc')
    
    plt.figure(3)
    plt.plot(sigma_range,MC_Result*np.ones([len(sigma_range),1]))
    plt.plot(sigma_range,L_T0_T1_T2+cc(sigma_range),'--r')
    plt.grid()
    plt.xlabel('sigma')
    plt.ylabel('value of derivative')
    plt.legend(['market price','price with cc'])
    
mainCalculation()
