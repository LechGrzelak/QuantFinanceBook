#%%
"""
Created on Wed Oct 16 2019
Exercise 4.1a- Newton Raphson algorithm
@author: Irfan Ilgin & Lech A. Grzelak
"""

from scipy.misc import derivative
import numpy as np
from scipy.optimize import brentq

# Function definition
f = lambda x: (np.exp(x) + np.exp(-x)) / 2 - (2 * x)

def combined_root_finder(func, interval, tol=1e-6, only_newton=False):
    if func(interval[0]) * func(interval[1]) > 0:
        print("there is no root in this interval!")
    elif func(interval[0]) * func(interval[1]) < 0:
        k = 1
        x = (interval[0] + interval[1]) / 2.0
        delta = - func(x) / derivative(func, x)
        while abs(delta / x) > tol:
            x = x + delta
            if not only_newton:
                if np.sum(x < interval) in [0, 2]:
                    if func(interval[0]) * func(x) > 0:
                        interval[0] = x - delta
                    else:
                        interval[1] = x - delta
                        x = (interval[0] + interval[1]) / 2.0
            delta = - func(x) / derivative(func, x)
            k += 1
            print(k, delta)
        return x, k
    else:
        print("You should buy a lottery ticket!")

# Combined root-finding
root_1, k1 = combined_root_finder(f, [2, 3])
root_2, k2 = combined_root_finder(f, [0, 2])

print('From combined root we have root = {0}'.format(root_1))
print('From combined root we have root = {0}'.format(root_2))

# Brent algorithm
brent_1 = brentq(f, 2, 3, full_output=True)
brent_2 = brentq(f, 0, 2, full_output=True)
print('From Brent algorithm we have root = {0}'.format(brent_1[0]))
print('From Brent algorithm we have root = {0}'.format(brent_2[0]))