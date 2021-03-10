#%%
"""
Created on Nov 14 2019
Exercise 8.11- the Bates model and the density recovery
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as st

def ChFBatesModel(r,tau,kappa,gamma,vbar,v0,rho,xiP,muJ,sigmaJ,S0):
    i = np.complex(0.0,1.0)
    D1 = lambda u: np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = lambda u: (kappa-gamma*rho*i*u-D1(u))/(kappa-gamma*rho*i*u+D1(u))
    C  = lambda u: (1.0-np.exp(-D1(u)*tau))/(gamma*gamma*(1.0-g(u)*\
                               np.exp(-D1(u)*tau)))*(kappa-gamma*rho*i*u-D1(u))
    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    AHes= lambda u: r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*\
        rho*i*u-D1(u)) - 2*kappa*vbar/gamma/gamma*np.log((1.0-g(u)*np.exp(-D1(u)*tau))/(1.0-g(u)))
    
    A = lambda u: AHes(u) - xiP * i * u * tau *(np.exp(muJ+0.5*sigmaJ*sigmaJ) - 1.0) + \
            xiP * tau * (np.exp(i*u*muJ - 0.5 * sigmaJ * sigmaJ * u * u) - 1.0)
    
    # Characteristic function for the Heston's model    
    cf = lambda u: np.exp(A(u) + C(u)*v0+ i*u*np.log(S0))
    return cf
def COSDensity(cf,x,N,a,b):
    i = np.complex(0.0,1.0) #assigning i=sqrt(-1)
    k = np.linspace(0,N-1,N)
    u = np.zeros([1,N])
    u = k * np.pi / (b-a)
        
    #F_k coefficients
    F_k    = 2.0 / (b - a) * np.real(cf(u) * np.exp(-i * u * a));
    F_k[0] = F_k[0] * 0.5; # adjustment for the first term
    
    #Final calculation
    f_X = np.matmul(F_k , np.cos(np.outer(u, x - a )))
        
    # we output only the first row
    return f_X

def mainCalculation():    
    # Heston model parameters
    v0    = 0.02
    S_0   = 100.0
    r     = 0.05
    T     = 5.0
    
    #define the range for the expansion points
    N = 4096
    L = 10
    
    kappa =0.5
    gamma =0.3
    vbar = 0.3
    rho = -0.7
    xiP = 0.2
    muJ = 0.1
    sigmaJ = 0.1
    
    
   # define the COS method integration range
    a = - L * np.sqrt(T)
    b = L * np.sqrt(T)

    # define domain for density
    x = np.linspace(-2,8,501)
    
    # Note that the ChF does not include the exp(iulog(S0)) term
    # this is a scaling term, important in the option pricing
    
    cf = ChFBatesModel(r,T,kappa,gamma,vbar,v0,rho,xiP,muJ,sigmaJ,S_0)
    f_Bates = COSDensity(cf,x,N,a,b) 
    
    plt.figure(1)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("$f_X(x)$")
    plt.plot(x,f_Bates)
    plt.legend(['pdf'])
    plt.title('PDF of the Bates model, log S(T)')
    integral = st.trapz(f_Bates,x)
    print('Integral over the density of f_X is equal to ={0}'.format(integral))
    
    y = np.exp(x)
    plt.figure(2)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("$f_S(x)$")
    plt.plot(y,1/y*f_Bates)
    plt.legend(['pdf'])
    plt.title('PDF of the Bates model, S(T)')
    integral = st.trapz(1/y*f_Bates,y)
    print('Integral over the density of f_S is equal to ={0}'.format(integral))
    ES = st.trapz(y*1/y*f_Bates,y)
    ES_Exact = S_0*np.exp(r*T)
    
    print('Expected stock, from density = {0}'.format(ES))
    print('Expected stock, exact = {0}'.format(ES_Exact))
    
    # Checking the effect of the model parameters on the density
    plt.figure(3)
    plt.title('effect of xiP on the tail of the density')
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("$log f_S(x)$")
    xiPV = [0.1, 0.2, 0.3, 0.5]
    leg = []
    for xiPTemp in xiPV:
        cf = ChFBatesModel(r,T,kappa,gamma,vbar,v0,rho,xiPTemp,muJ,sigmaJ,S_0)
        f_Bates = COSDensity(cf,x,N,a,b)
        plt.plot(y,np.log(1/y*f_Bates))
        leg.append('xiP={0}'.format(xiPTemp))
    plt.legend(leg)
    
    plt.figure(4)
    plt.title('effect of muJ on the tail of the density')
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("$log f_S(x)$")
    muJV = [-0.3,-0.2,0.0, 0.1, 0.2, 0.3,0.5]
    leg = []
    for muJTemp in muJV:
        cf = ChFBatesModel(r,T,kappa,gamma,vbar,v0,rho,xiP,muJTemp,sigmaJ,S_0)
        f_Bates = COSDensity(cf,x,N,a,b)
        plt.plot(y,np.log(1/y*f_Bates))
        leg.append('muJ={0}'.format(muJTemp))
    plt.legend(leg)
    
    plt.figure(5)
    plt.title('effect of sigmaJ on the tail of the density')
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("$log f_S(x)$")
    sigmaJV = [0.1, 0.2, 0.3, 0.5]
    leg = []
    for sigmaJTemp in sigmaJV:
        cf = ChFBatesModel(r,T,kappa,gamma,vbar,v0,rho,xiP,muJ,sigmaJTemp,S_0)
        f_Bates = COSDensity(cf,x,N,a,b)
        plt.plot(y,np.log(1/y*f_Bates))
        leg.append('sigmaJ={0}'.format(sigmaJTemp))
    plt.legend(leg)
mainCalculation()