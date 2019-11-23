#%%
"""
Created on Feb 11 2019
Local volatility model based on the implied volatility obtained from Hagan's formula
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

def HaganImpliedVolatility(K,T,f,alpha,beta,rho,gamma):

    # We make sure that the input is of array type

    if type(K) == float:
        K = np.array([K])
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    
    # The strike prices cannot be too close to 0

    K[K<1e-10] = 1e-10
    
    z        = gamma/alpha*np.power(f*K,(1.0-beta)/2.0)*np.log(f/K)
    x_z      = np.log((np.sqrt(1.0-2.0*rho*z+z*z)+z-rho)/(1.0-rho))
    A        = alpha/(np.power(f*K,((1.0-beta)/2.0))*(1.0+np.power(1.0-beta,2.0)/24.0*
                               np.power(np.log(f/K),2.0)+np.power((1.0-beta),4.0)/1920.0*
                               np.power(np.log(f/K),4.0)))
    B1       = 1.0 + (np.power((1.0-beta),2.0)/24.0*alpha*alpha/(np.power((f*K),
                1-beta))+1/4*(rho*beta*gamma*alpha)/(np.power((f*K),
                             ((1.0-beta)/2.0)))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T
    impVol   = A*(z/x_z) * B1
    B2 = 1.0 + (np.power(1.0-beta,2.0)/24.0*alpha*alpha/
                (np.power(f,2.0-2.0*beta))+1.0/4.0*(rho*beta*gamma*
                alpha)/np.power(f,(1.0-beta))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T

    # Special treatment of ATM strike value

    impVol[np.where(K==f)] = alpha / np.power(f,(1-beta)) * B2
    return impVol

def LocalVarianceBasedOnSABR(s0,frwd,r,alpha,beta,rho,volvol):

    # Define shock size for approximating derivatives

    dt =0.001
    dx =0.001

    # Function for Hagan's implied volatility approximation 

    sigma =lambda x,t: HaganImpliedVolatility(x,t,frwd,alpha,beta,rho,volvol)

    # Derivatives

    dsigmadt   = lambda x,t: (sigma(x,t+dt)-sigma(x,t))/dt
    dsigmadx   = lambda x,t: (sigma(x+dx,t)-sigma(x-dx,t))/(2.0*dx)
    d2sigmadx2 = lambda x,t: (sigma(x+dx,t) + sigma(x-dx,t)-2.0*sigma(x,t))/(dx*dx)
    omega      = lambda x,t: sigma(x,t)*sigma(x,t)*t
    domegadt   = lambda x,t: sigma(x,t)**2.0 + 2.0*t*sigma(x,t)*dsigmadt(x,t)
    domegadx   = lambda x,t: 2.0*t*sigma(x,t)*dsigmadx(x,t)
    #d2omegadx2 = lambda x,t: 2.0*t*(dsigmadx(x,t))**2.0 + 2.0*t*sigma(x,t)*d2sigmadx2(x,t)
    d2omegadx2 = lambda x,t: 2.0*t*np.power(dsigmadx(x,t),2.0) + 2.0*t*sigma(x,t)*d2sigmadx2(x,t)

    term1    = lambda x,t:  1.0+x*domegadx(x,t)*(0.5 - np.log(x/(s0*np.exp(r*t))) / omega(x,t))    
    term2    = lambda x,t:  0.5*np.power(x,2.0)*d2omegadx2(x,t)
    term3    = lambda x,t:  0.5*np.power(x,2.0)*np.power(domegadx(x,t),2.0)*(-1.0/8.0-1.0/(2.0*omega(x,t))\
            +np.log(x/(s0*np.exp(r*t)))*np.log(x/(s0*np.exp(r*t)))/(2*omega(x,t)*omega(x,t)))

    # Final expression for local variance

    sigmalv2 = lambda x,t:(domegadt(x,t)+r*x*domegadx(x,t))/(term1(x,t)+term2(x,t)+term3(x,t))
    return sigmalv2
 
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

def EUOptionPriceFromMCPaths(CP,S,K,T,r):

    # S is a vector of Monte Carlo samples at T

    if CP == OptionType.CALL:
        return np.exp(-r*T)*np.mean(np.maximum(S-K,0.0))
    elif CP == OptionType.PUT:
        return np.exp(-r*T)*np.mean(np.maximum(K-S,0.0))
    
def mainCalculation():

    # For the SABR model we take beta =1 and rho =0 (as simplification)

    beta   = 1.0
    rho    = 0.0

    # Other model parameters

    volvol = 0.2
    s0     = 1.0
    T      = 10.0
    r      = 0.05
    alpha  = 0.2
    f_0    = s0*np.exp(r*T)
    CP     = OptionType.CALL

    # Monte Carlo settings

    NoOfPaths = 25000
    NoOfSteps = (int)(100*T)
    
    # We define the market to be driven by Hagan's SABR formula
    # Based on this formula we derive the local volatility/variance 

    sigma     = lambda x,t: HaganImpliedVolatility(x,t,f_0,alpha,beta,rho,volvol)
    
    # Local variance based on the Hagan's SABR formula

    sigmalv2  = LocalVarianceBasedOnSABR(s0,f_0,r,alpha,beta,rho,volvol)
           
    # Monte Carlo simulation

    dt        = T/NoOfSteps
    np.random.seed(4)
    Z         = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    S         = np.zeros([NoOfPaths,NoOfSteps+1])
    
    S[:,0]    = s0;
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
        temp = sigmalv2(S_i,time[i])
        sig = np.real(temp)
        np.nan_to_num(sig)
        
        # Because of discretizations we may encouter negative variance which
        # is set to 0 here.

        sig=np.maximum(sig,1e-14)
        sigmaLV = np.sqrt(sig)  
        
        # Stock path

        S[:,i+1]=S[:,i] * (1.0 + r*dt + sigmaLV.transpose()*Z[:,i]*np.sqrt(dt))
                       
        # We force that at each time S(t)/M(t) is a martingale

        S[:,i+1]= S[:,i+1] - np.mean(S[:,i+1]) + s0*np.exp(r*time[i])
        
        # Make sure that after moment matching we don't encounter negative stock values

        S[:,i+1]=np.maximum(S[:,i+1],1e-14)
        
        # Adjust time

        time[i+1] = time[i] + dt

    # Plot some results

    K = np.linspace(0.2,5.0,25)
    #c_n = np.array([-1.5, -1.0, -0.5,0.0, 0.5, 1.0, 1.5])
    #K= s0*np.exp(r*T) * np.exp(0.1 * c_n * np.sqrt(T))
    OptPrice = np.zeros([len(K),1])
    IV_Hagan = np.zeros([len(K),1])
    IV_MC = np.zeros([len(K),1])
    for (idx,k) in enumerate(K):
        OptPrice[idx] = EUOptionPriceFromMCPaths(CP,S[:,-1],k,T,r)
        IV_Hagan[idx] = sigma([k],T)*100.0
        IV_MC[idx]    = ImpliedVolatility(CP,OptPrice[idx],k,T,s0,r)*100.0
        
    # Plot the option prices

    plt.figure(1)
    plt.plot(K,OptPrice)
    plt.grid()
    plt.xlabel('strike')
    plt.ylabel('option price')
    
    # Plot the implied volatilities

    plt.figure(2)
    plt.plot(K,IV_Hagan)
    plt.plot(K,IV_MC,'-r')
    plt.grid()
    plt.xlabel('strike')
    plt.ylabel('implied volatility')
    plt.legend(['Hagan','Monte Carlo'])
    plt.axis([np.min(K),np.max(K),0,40])
mainCalculation()
