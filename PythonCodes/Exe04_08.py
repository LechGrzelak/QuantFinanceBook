#%%
"""
Created on Thu Nov 15 2018
Implied volatilities and densities obtained from Hagan's formula
@author: Lech A. Grzelak
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):

    # Black-Scholes call option price
    # We make sure that the input is of array type

    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
        * tau) / (sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if str(CP).lower()=="c" or str(CP).lower()=="1":
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif str(CP).lower()=="p" or str(CP).lower()=="-1":
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

def HaganImpliedVolatility(K,T,f,alpha,beta,rho,gamma):

    # We make sure that the input is of array type

    if type(K) == float:
        K = np.array([K])
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])
    z        = gamma/alpha*np.power(f*K,(1.0-beta)/2.0)*np.log(f/K);
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
                alpha)/np.power(f,(1.0-beta))+(2.0-3.0*rho*rho)/24.0*gamma*gamma)*T;

    # Special treatment of ATM strike price

    impVol[np.where(K==f)] = alpha / np.power(f,(1-beta)) * B2;
    return impVol

def DVdK(x,CP,beta,alpha,rho,gamma,f_0,T,r,bump):

    # We make sure that the input is of array type

    if x is not np.array:
        x = np.array(x).reshape([len(x),1])
    iv_f = lambda x : HaganImpliedVolatility(x,T,f_0,alpha,beta,rho,gamma)
    optValue = lambda x: BS_Call_Option_Price(CP,f_0,x,iv_f(x),T,r)
    DV_dK_lambda = lambda x: (optValue(x+bump)-optValue(x-bump))/(2.0 * bump)
    return DV_dK_lambda(x)

def D2VdK2(x,CP,beta,alpha,rho,gamma,f_0,T,r,bump):

    # We make sure that the input is of array type

    if x is not np.array:
        x = np.array(x).reshape([len(x),1])
    iv_f       = lambda x: HaganImpliedVolatility(x,T,f_0,alpha,beta,rho,gamma)
    optValue   = lambda x: BS_Call_Option_Price(CP,f_0,x,iv_f(x),T,r)
    D2VdK2_lambda = lambda x: (optValue(x+bump) + optValue(x-bump) - 
                               2.0 * optValue(x))/(bump*bump)
    return D2VdK2_lambda(x)

def mainCalculation():

    # Parameters for the SABR approximation formula

    CP    = 1
    beta  = 0.5
    alpha = 0.2
    rho   = -0.7
    gamma = 0.35
    S0    = 5.6
    T     = 2.5
    r     = 0.015
    SF0   = S0 *np.exp(r*T)

    # Shock size used in finite differences calculations of derivatives

    bump      = 0.00001
    
    # Figure 1 dVdK

    plt.figure(1)
    x     = np.linspace(0.0001,2.0*SF0,100)
    dVdK  = DVdK(x,CP,beta,alpha,rho,gamma,SF0,T,r,bump)
    plt.plot(x, dVdK)
    plt.xlabel('x')
    plt.ylabel('dVdx')
    plt.grid()
    
    # Figure 2 d2VdK2

    plt.figure(2)
    d2Vdx2  = D2VdK2(x,CP,beta,alpha,rho,gamma,SF0,T,r,bump)
    plt.plot(x, d2Vdx2)
    plt.xlabel('x')
    plt.ylabel('d2Vdx2')
    plt.grid()
    
    # Density is now equal to

    f_x = d2Vdx2 * np.exp(r*T)
    
    # Integrated density 

    I = np.trapz(np.squeeze(f_x),x)

    # Integrate the density and check integration to 0

    print('Integrated density is equal to = {0}'.format(I))

mainCalculation()
