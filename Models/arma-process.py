'''
Compute the ARMA Process Variance, ACOVF and ACF.
'''


import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import acovf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# Zt = phi[1]*Zt_1 + at + theta[1]*at_1
phi = [] # <== AR parameters
theta = [] # <== MA parameters
k = 1 # . . . . . . Lag for the printed results
noise_mu = 0.0 #. . Mean of the random noise
noise_var = 1.0 # . Variance of the white noise
N = 100 # . . . . Number of simulated samples if 0 the program will simply output the metrics
show_plot = True # . . . . If to display the plot
save_picture = False #. . . If you want to save the pictures for the plot


p = len(phi) #. . Dimension of the model
q = len(theta)
theta = [1.0] + theta
phi = [1.0] + phi 
initial_state = 0.0 # . . . Initial state of the model
realizations = [] # . . . . Model realizations
plot_index = 0 #. . 0 = plot the realizations
# . . . . . . . . . 1 = plot the ACF up to k
# . . . . . . . . . 2 = plot the PACF up to k


def gamma_f(k):
    global phi
    global theta
    global noise_var
    if k == 0:
        global var
        return var
    if k == 1:
        return ( (phi[1] + theta[1])*(1 + phi[1]*theta[1]) ) / (1 - phi[1]**2) * noise_var
    else:
        return phi[1]*gamma_f(k-1)

def rho_f(k):
    global phi
    global theta

    if k == 0:
        return 1.0
    elif k == 1:
        return gamma_f(k) / gamma_f(0)
    else:
        return phi[1]*rho_f(k-1)

def model():
    '''
    The model that will generate the time series.
    Supports up to three parameters.
    '''
    # load parameters
    global phi
    global p
    if p >= 1:
        phi1 = phi[1]
    else:   
        phi1 = 0.0
    if p >= 2:
        phi2 = phi[2]
    else:
        phi2 = 0.0
    
    # load past variables state
    global at
    global Zt
    global Zt_1
    global at_1

    # update state
    at_1 = at
    Zt_1 = Zt

    # get noise
    global noise_mu
    global noise_var
    at = np.random.normal(noise_mu, noise_var)

    # compute current state
    
    Zt = phi[1]*Zt_1 + at + theta[1]*at_1

    return Zt


def run_simulation(N):
    '''
    Run N steps of the simulation and stores it in the realizations list.
    '''
    global realizations
    for i in range(N):
        realizations.append(model())


Zt   = initial_state
Zt_1 = initial_state
Zt_2 = initial_state
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
if p == 1 and q == 1:
    var = ( 1 + theta[1]**2 + 2*phi[1]*theta[1]) / (1 - phi[1]**2) * noise_var
else:
    print("Not supported")
    assert(False)

gamma = gamma_f(k)

if p == 1 and q == 1:
    rho = rho_f(k)
else:
    print("p > 2 is not supported")
    assert(False)

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