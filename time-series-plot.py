'''
Simulate and reason over a simple random process.
'''

import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import acovf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


#  if empty it will simulate the model expressed in the parameters below
realizations = [0.38, 0.64, -0.91, -0.74, -0.34, -0.36, 1.00, 0.51, 0.52, -0.91, -0.29, -0.87, 0.64, 0.98, -0.74] # <== copy the realizations here
k = 1   # . . . . . . . . . Lag for the ATF and PACF functions

N = 100 # . . . . . . . . . Number of simulated samples
initial_state = 0.0 # . . . Initial state of the model
w1 = +0.5 # . . . . . . . . Weight for Zt_1
w2 = +0.0 # . . . . . . . . Weight for Zt_2
w3 = +0.0 # . . . . . . . . Weight for Zt_3

show_plot = False # . . . . If to display the plot
save_picture = False #. . . If you want to save the pictures for the plot
plot_index = 0 #. . 0 = plot the realizations
# . . . . . . . . . 1 = plot the ACF up to lag
# . . . . . . . . . 2 = plot the PACF up to lag


def model(w1=0.0, w2=0.0, w3=0.0):
    '''
    The model that will generate the time series.
    Supports up to three parameters.
    '''
    # load past variables state
    global Zt
    global Zt_1
    global Zt_2
    global Zt_3

    # update state
    Zt_3 = Zt_2
    Zt_2 = Zt_1
    Zt_1 = Zt

    # get noise
    at = np.random.normal(0.0, 1.0)

    # compute current state
    Zt = w1*Zt_1 + w2*Zt_2 + w3*Zt_3 + at

    return Zt


def run_simulation(N, w1=0.0, w2=0.0, w3=0.0):
    '''
    Run N steps of the simulation and stores it in the realizations list.
    '''
    global realizations
    realizations = []
    for i in range(N):
        realizations.append(model(w1, w2, w3))


Zt   = initial_state
Zt_1 = initial_state
Zt_2 = initial_state
Zt_3 = initial_state

if len(realizations) > 0:
    N = len(realizations)

if k < 0 and plot_index != 0:
    k = N-1 if plot_index == 1 else int((N/2)-1)
elif plot_index == 1 and k >= N:
    k = N-1
elif plot_index == 2 and k >= (N/2):
    k = int((N/2)-1)
elif k < 0 and plot_index == 0:
    k = 0

if len(realizations) == 0:
    run_simulation(N, w1, w2, w3)

realizations = np.array(realizations)
mu = np.mean(realizations)
std = np.std(realizations)
variance = np.var(realizations)
autocorrelation = acf(realizations)[min(k,len(acf(realizations))-1)]
autocovariance = acovf(realizations)[k]
print("\nMean:_______{mean}\nDeviation:__{std}\nVariance:___{var}\n\nAutocovariance (gamma):__{autocov}\nAutocorrelation (rho):___{acf}\n".format(mean=mu, std=std, var=variance, autocov=autocovariance, acf=autocorrelation))

# plotting the points
assert(plot_index >= 0 and plot_index <= 2)
plots = ["time-series", "ACF", "PACF" ]
plot_type = plots[plot_index]

if plot_type == "ACF":
    plot_acf(realizations, lags=k)
elif plot_type == "PACF":
    plot_pacf(realizations, lags=k)
elif plot_type == "time-series":
    plt.plot(realizations)
    plt.axhline(y=mu, linestyle='dotted')

# naming the x axis 
plt.xlabel('time step') 
# naming the y axis 
plt.ylabel(plot_type)
# giving a title to my graph 
plt.title(plot_type + " plot")
  
# function to show the plot 
if save_picture:
    plt.savefig("{title}-plot.png".format(title=plot_type))
if show_plot:
    plt.show()