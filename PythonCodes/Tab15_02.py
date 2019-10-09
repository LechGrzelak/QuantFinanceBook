#%%
"""
Created on Feb 10 2019
Strikes for for the FX options
@author: Lech A. Grzelak
"""
import numpy as np

def GenerateStrikes(frwd,Ti):
    c_n = np.array([-1.5, -1.0, -0.5,0.0, 0.5, 1.0, 1.5])
    return frwd * np.exp(0.1 * c_n * np.sqrt(Ti))
   
def mainCalculation():
    TiV = [0.5, 1.0, 5.0, 10.0, 20.0, 30.0]
    strikesM = np.zeros([len(TiV),7])
    
    # Market ZCBs for domestic and foreign markets

    P0T_d = lambda t: np.exp(-0.02 * t)
    P0T_f = lambda t: np.exp(-0.05 * t)
    
    # Spot of the FX rate

    y0 = 1.35
    
    for (idx,ti) in enumerate(TiV):
        frwd = y0 * P0T_f(ti) / P0T_d(ti)
        strikesM[idx,:] = GenerateStrikes(frwd,ti)
       
    print(strikesM)
        
mainCalculation()
