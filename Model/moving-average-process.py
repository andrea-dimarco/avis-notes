'''
Compute the Moving Average Process Variance, ACOVF and ACF.
'''


import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import acovf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# Write the MA model parameters down here
theta = [-0.5]
k = 1 # . . . . . . Lag for the printed results
noise_mu = 0.0 #. . Mean of the random noise
noise_var = 1.0 # . Variance of the white noise
N = 1000 # . . . . Number of simulated samples if 0 the program will simply output the metrics
show_plot = False # . . . . If to display the plot
save_picture = False #. . . If you want to save the pictures for the plot


q = len(theta) #. . Dimension of the model
theta = [1.0] + theta # DO NOT TOUCH THIS
initial_state = 0.0 # . . . Initial state of the model
realizations = [] # . . . . Model realizations
plot_index = 0 #. . 0 = plot the realizations
# . . . . . . . . . 1 = plot the ACF up to k
# . . . . . . . . . 2 = plot the PACF up to k


def model():
    '''
    The model that will generate the time series.
    Supports up to three parameters.
    '''
    # load parameters
    global theta

    if q >= 1:
        theta1 = theta[1]
    else:   
        theta1 = 0.0
    if q >= 2:
        theta2 = theta[2]
    else:
        theta2 = 0.0
    
    # load past variables state
    global at
    global at_1
    global at_2

    # update state
    at_2 = at_1
    at_1 = at

    # get noise
    at = np.random.normal(noise_mu, noise_var)

    # compute current state
    
    Zt = at + theta1*at_1 + theta2*at_2

    return Zt


def run_simulation(N):
    '''
    Run N steps of the simulation and stores it in the realizations list.
    '''
    global realizations
    for i in range(N):
        realizations.append(model())


at_1 = noise_mu
at_2 = noise_mu
at   = noise_mu

if k < 0:
    k = N-1 if plot_index == 1 else (N/2)-1
elif plot_index == 1 and k >= N:
    k = N-1
elif plot_index == 2 and k >= (N/2):
    k = int((N/2)-1)

if N > 0:
    run_simulation(N)
    realizations = np.array(realizations)
    simulated_var = np.var(realizations)
    simulated_mu = np.mean(realizations)
    simulated_gamma = acovf(realizations)[k]
    simulated_rho = acf(realizations)[k]
    print("Simulated results:\n Mean:____________________{mu}\n Variance:________________{var}\n Autocovariance (gamma):__{gamma}\n Autocorrelation (rho):___{rho}\n".format(var=simulated_var, mu=simulated_mu, gamma=simulated_gamma, rho=simulated_rho))

var = 0
for i in range(q+1):
    var += theta[i]**2
var *= noise_var

gamma = 0.0
for s in range(q-k+1):
    gamma += theta[s]*theta[s+k]
gamma *= noise_var

rho = gamma / noise_var / var

print("Computed results\n Variance:________________{var}\n Autocovariance (gamma):__{gamma}\n Autocorrelation (rho):___{rho}\n".format(var=var, gamma=gamma, rho=rho))

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