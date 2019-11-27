#%%
"""
Created on Wed Oct 16 2019
Exercise 3.3- Strangle payoff
@author: Irfan Ilgin & Lech A. Grzelak
"""
import matplotlib.pyplot as plt
import numpy as np


delta_t = 0.5

S       = np.linspace(0, 40, num=400)

K1      = 20
K2      = 10
payoff  = np.maximum(S - K1, 0) + np.maximum(K2 - S, 0)

plt.plot(S, payoff)
plt.grid()
plt.xlabel('S')
plt.ylabel('H(T, S)')
plt.show()
