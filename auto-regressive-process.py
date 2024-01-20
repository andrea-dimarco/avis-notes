'''
Compute the Auto-Regressive Process Variance, ACOVF and ACF.
'''


import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import acovf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


phi = [0.5] # Model parameters
k = 1 # . . . . . . Lag for the printed results
noise_mu = 0.0 #. . Mean of the random noise
noise_var = 1.0 # . Variance of the white noise
N = 100000 # . . . . Number of simulated samples if 0 the program will simply output the metrics
show_plot = False # . . . . If to display the plot
save_picture = False #. . . If you want to save the pictures for the plot


p = len(phi) #. . Dimension of the model
phi = [1.0] + phi # DO NOT TOUCH THIS
initial_state = 0.0 # . . . Initial state of the model
realizations = [] # . . . . Model realizations
plot_index = 0 #. . 0 = plot the realizations
# . . . . . . . . . 1 = plot the ACF up to k
# . . . . . . . . . 2 = plot the PACF up to k


def gamma_f(k):
    global p
    global phi
    if k == 0:
        global var
        return var
    else:
        sum = 0
        for i in range(1, p+1):
            sum += phi[i]*gamma_f(k-i)
        return sum

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
    global Zt_2

    # update state
    Zt_2 = Zt_1
    Zt_1 = Zt

    # get noise
    at = np.random.normal(noise_mu, noise_var)

    # compute current state
    
    Zt = at + phi1*Zt_1 + phi2*Zt_2

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
if p == 1:
    var = noise_var / (1 - phi[1]**2)
elif p == 2:
    var = ( (1-phi[2])*noise_var ) / ( (1+phi[2])*(1-phi[1]-phi[2])*(1+phi[1]-phi[2]) )
else:
    print("Not supported")
    assert(False)

gamma = gamma_f(k)

if p == 1:
    rho = phi[1]**k
elif p == 2:
    rho = gamma_f(k) / gamma_f(0)
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