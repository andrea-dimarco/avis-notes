'''
Solve the model as difference equation.
'''

import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import acovf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

#             Zt-3  Zt-2  Zt-1  Zt=1
polynomial = [+0.0, +0.3, -1.1, +1.0]
inverse_roots = np.roots(polynomial)
roots = [1/x for x in inverse_roots]

print("Roots: {r}".format(r=roots))

alpha = 0.0
phi = 0.0

for root in roots:
    if root.imag != 0.0:
        d = root.imag
        c = root.real
        alpha = np.sqrt((c**2) + (d**2))
        phi = np.abs(np.arctan( d / c ))
        phi2 = np.pi / phi
        print(" Alpha: {a}\n Phi:   {phi} = pi / {phi2}".format(a=alpha, phi=phi, phi2=phi2))
        break;