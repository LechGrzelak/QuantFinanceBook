#%%
"""
CIR distribution
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Global variables 

s = 0
t = 5
x = np.linspace(0.001,0.501,100)

# Feller condition

def FellerCondition(kappa,v_bar,gamma):
    q_F = 2.0*kappa*v_bar/np.power(gamma,2.0)-1.0
    print("Feller condition, q_F is equal to %.2f" %q_F)
    return q_F
        
# PDF and CDF for CIR process

def CIR_PDF_CDF(x,gamma,kappa,t,s,v_bar,v_s):    
    c_s_t      = np.power(gamma,2.0) / (4.0*kappa) * (1.0-np.exp(-kappa*(t-s)));
    d          = 4.0 * kappa * v_bar / np.power(gamma,2.0);
    lambda_s_t = 4.0 * kappa * np.exp(-kappa * (t-s)) / (np.power(gamma,2.0) * (1.0 - np.exp(-kappa * (t-s)))) * v_s;
    f_X        = 1.0 / c_s_t * st.ncx2.pdf(x/c_s_t,d,lambda_s_t);
    F_X        = st.ncx2.cdf(x/c_s_t,d,lambda_s_t);
    output = {"pdf":f_X, "cdf":F_X}    
    return output

def Analysis():

    # Feller condition satisfied

    gamma = 0.316
    kappa = 0.5
    v_s   = 0.2
    v_bar = 0.05
    q_F = FellerCondition(kappa,v_bar,gamma)
    print(q_F)
    output_case1 = CIR_PDF_CDF(x,gamma,kappa,t,s,v_bar,v_s)
    
    # Feller condition not satisfied

    gamma    = 0.129
    kappa    = 0.5
    v_s      = 0.2
    v_bar    = 0.05
    output_case2 = CIR_PDF_CDF(x,gamma,kappa,t,s,v_bar,v_s)
    
    # Generate output plots

    plt.figure(1)
    plt.plot(x,output_case1["pdf"])
    plt.plot(x,output_case2["pdf"])
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim([0,15])
    plt.figure(2)
    plt.plot(x,output_case1["cdf"])
    plt.plot(x,output_case2["cdf"])
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("F(x)")

Analysis()
