#%%
"""
Created on Wed Oct 16 2019
Exercise 4.9- density computed from interpolated implied volatilities
@author: Irfan Ilgin & Lech A. Grzelak
"""
import numpy as np
from math import exp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.stats as st

def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):
    # Black-Scholes call option price
    d1    = (np.log(S_0 / float(K)) + (r + 0.5 * np.power(sigma,2.0)) * tau) / float(sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if str(CP).lower()=="c" or str(CP).lower()=="1":
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif str(CP).lower()=="p" or str(CP).lower()=="-1":
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value


def scipy_1d_interpolate(xs, ys, kind):
    strikes = np.linspace(xs[0], xs[-1], (xs[-1] - xs[0]) * 4)
    imp_vol = interp1d(xs, ys, kind=kind, fill_value=(ys[0], ys[-1]), bounds_error=False)(strikes)
    return imp_vol, strikes

def pdf_from_vc(imp_vols, Ks, S0, T, r, t):
    pdf = []
    for i in range(1, len(imp_vols)-1):
        y_back    = BS_Call_Option_Price("c",S0,Ks[i-1],imp_vols[i-1],T-t,r)
        y_forward = BS_Call_Option_Price("c",S0,Ks[i+1],imp_vols[i+1],T-t,r)
        y         = BS_Call_Option_Price("c",S0,Ks[i],imp_vols[i],T-t,r)
        pdf_i     = (y_forward + y_back - 2 * y) / ((-Ks[i-1] + Ks[i+1])/2)**2
        pdf.append(pdf_i)
    pdf = np.array(pdf) * exp(r * (T-t))
    return pdf

K = np.array([3.28, 5.46, 8.2, 10.93, 13.66, 16.39, 19.12, 21.86])
imp_vol = np.array([0.3137, 0.2249, 0.1491, 0.0909, 0.0685, 0.0809, 0.0945, 0.1063])

vol_f = lambda x: np.interp(x,K,imp_vol,left=imp_vol[0], right=imp_vol[-1])

# Market settings
S0 = 10.5
r  = 0.04
T  = 1
t  = 0

plt.figure(1)
plt.grid()
impvol_lin, K_lin = scipy_1d_interpolate(K, imp_vol, "slinear")
pdf = pdf_from_vc(impvol_lin, K_lin, S0, T, r, t)
plt.plot(K_lin[1:-1], pdf)
plt.plot(K_lin, impvol_lin)
plt.xlabel("K")
plt.ylabel("pdf")
plt.legend(["pdf","IV"])

plt.figure(2)
plt.grid()
impvol_cubic, K_cubic = scipy_1d_interpolate(K, imp_vol, "cubic")
pdf = pdf_from_vc(impvol_cubic, K_cubic, S0, T, r, t)
plt.plot(K_cubic[1:-1], pdf)
plt.plot(K_cubic, impvol_cubic)
plt.xlabel("K")
plt.ylabel("pdf")
plt.legend(["pdf","IV"])

plt.figure(3)
plt.grid()
impvol_nearest, K_nearest = scipy_1d_interpolate(K, imp_vol, "nearest")
pdf = pdf_from_vc(impvol_nearest, K_nearest, S0, T, r, t)
plt.plot(K_nearest[1:-1], pdf)
plt.plot(K_nearest, impvol_nearest)
plt.xlabel("K")
plt.ylabel("pdf")
plt.legend(["pdf","IV"])
