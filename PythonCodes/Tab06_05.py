#%%
"""
Created on Thu Jan 16 2019
Pricing of Cash-or-Nothing options with the COS method
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import time

def CashOrNothingPriceCOSMthd(cf,CP,S0,r,tau,K,N,L):


    # cf   - Characteristic function is a function, in the book denoted by \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # r    - Interest rate (constant)
    # tau  - Time to maturity
    # K    - List of strikes
    # N    - Number of expansion terms
    # L    - Size of truncation domain (typ.:L=8 or L=10)

    # Reshape K to become a column vector

    K = np.array(K).reshape([len(K),1])
    
    # Assigning i=sqrt(-1)

    i = np.complex(0.0,1.0) 
    
    x0 = np.log(S0 / K)   
    
    # Truncation domain

    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # Summation from k = 0 to k=N-1

    k = np.linspace(0,N-1,N).reshape([N,1])  
    u = k * np.pi / (b - a);  

    # Determine coefficients for put prices  

    H_k = CashOrNothingCoefficients(CP,a,b,k)#CallPutCoefficients(CP,a,b,k)
       
    mat = np.exp(i * np.outer((x0 - a) , u))

    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    
    value = np.exp(-r * tau) *K * np.real(mat.dot(temp))
         
    return value

""" 
Determine coefficients for cash or nothing option
"""
def CashOrNothingCoefficients(CP,a,b,k):
    if str(CP).lower()=="c" or str(CP).lower()=="1":                  
        c = 0.0
        d = b
        coef = Chi_Psi(a,b,c,d,k)
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k),1])
        else:
            H_k      = 2.0 / (b - a) * (Psi_k)  
        
    elif str(CP).lower()=="p" or str(CP).lower()=="-1":
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        Psi_k = coef["psi"]
        H_k      = 2.0 / (b - a) * (Psi_k)               
    
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
    

def BS_Cash_Or_Nothing_Price(CP,S_0,K,sigma,tau,r):

    # Black-Scholes call option price

    cp = str(CP).lower()
    K = np.array(K).reshape([len(K),1])
    d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) 
    * tau) / float(sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if cp == "c" or cp == "1":
        value = K * np.exp(-r * tau) * st.norm.cdf(d2)
    elif cp == "p" or cp =="-1":
        value = K * np.exp(-r * tau) *(1.0 - st.norm.cdf(d2))
    return value

def mainCalculation():
    i = np.complex(0.0,1.0)
    
    CP    = "p"
    S0    = 100.0
    r     = 0.05
    tau   = 0.1
    sigma = 0.2
    K     = [120] #np.linspace(10,S0*2.0,50)#[120.0]#
    N     = [40, 60, 80, 100, 120, 140]
    L     = 6
    
    # Definition of the characteristic function for GBM, this is an input
    # for the COS method
    # Note that Chf does not include "+iuX(t_0)" this coefficient
    # is included internally in the evaluation
    # In the book we denote this function by \varphi(u)

    cf = lambda u: np.exp((r - 0.5 * np.power(sigma,2.0)) * i * u * tau - 0.5 
                          * np.power(sigma, 2.0) * np.power(u, 2.0) * tau)
    
    val_COS_Exact = CashOrNothingPriceCOSMthd(cf,CP,S0,r,tau,K,np.power(2,14),L);
    print("Reference value is equal to ={0}".format(val_COS_Exact[0][0]))

    # Timing results 

    NoOfIterations = 1000

    for n in N:
        time_start = time.time() 
        for k in range(0,NoOfIterations):
            val_COS = CashOrNothingPriceCOSMthd(cf,CP,S0,r,tau,K,n,L)[0]
        print("For N={0} the error is ={1}".format(n,val_COS[0]-val_COS_Exact[0]))
        time_stop = time.time()
        print("It took {0} seconds to price.".format((time_stop-time_start)/float(NoOfIterations)))
        
        
mainCalculation()
