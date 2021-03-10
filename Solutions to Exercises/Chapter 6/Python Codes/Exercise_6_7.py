#%%
"""
Created on Nov 09 2019
Cumulants for the CGMY model using symbolic computation in Python
@author: Irfan Ilgin and Lech A. Grzelak
"""

from sympy import symbols, diff, simplify
from sympy.functions.elementary.exponential import exp, log

C, G, M, Y, t, r, s, u, A = symbols("C, G, M, Y, t, r, s, u, A")

def char_cgmy(C, G, M, Y, t, brownian=False, r=None, sigma=None):
    phi_cgmy = lambda u: exp(C * t * A * (
            (M - 1j * u) ** Y - M ** Y + (G + 1j * u) ** Y - G ** Y))
    if not brownian:
        return phi_cgmy
    elif brownian:
        omega = -1 * C * A * (
                (M - 1)** Y - M ** Y + (G + 1) ** Y - G ** Y)
        mu = r - 0.5 * sigma ** 2 + omega
        phi = lambda u: phi_cgmy(u) * exp(
            1j * u * mu * t - 0.5 * t * (sigma * u)** 2)
        return phi


g = char_cgmy(C, G, M, Y, t, brownian=True, r=r, sigma=s)

cumulant1 = diff(log(g(u))/1j, u)
cumulant2 = diff(cumulant1/1j, u)
cumulant3 = diff(cumulant2/1j, u)
cumulant4 = diff(cumulant3/1j, u)

print(simplify(cumulant1.subs(u,0)))
print(simplify(cumulant2.subs(u,0)))
print(simplify(cumulant3.subs(u,0)))
print(simplify(cumulant4.subs(u,0)))


print(diff((M-1j*u)**Y, u).subs(u, 0)/1j)