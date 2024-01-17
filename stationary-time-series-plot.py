import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


N = 50  # . . . . . . . . . Number of samples
k = 0 # . . . . . . . . . . ACF and PACF horizon, if 0 it will be the maximum amount
initial_state = 0.0 # . . . Initial state of the model
w1 = -1.2 # . . . . . . . . Weight for Zt_1
w2 = +0.8 # . . . . . . . . Weight for Zt_2
w3 = +0.0 # . . . . . . . . Weight for Zt_3
compute_function = True # . If to compute the function or write the realizations on your own
save_picture = True # . . . If you want to save the pictures for the plot
# . . . . . . . . . . . . . Write realizations manually here (not a smart idea)
realizations = [53, 43, 66, 48, 52, 42, 44, 56, 44, 58, 41, 54, 51, 56, 38, 56, 49, 52, 32, 52, 59, 34, 57, 39, 60, 40, 52, 44, 65, 43] 

plot_index = 2 #. . 0 = plot the realizations
# . . . . . . . . . 1 = plot the ACF up to k
# . . . . . . . . . 2 = plot the PACF up to k


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

if k <= 0:
    k = N-1 if plot_index == 1 else (N/2)-1
if compute_function:
    run_simulation(N, w1, w2, w3)
realizations = np.array(realizations)
mu = np.mean(realizations)
std = np.std(realizations)
variance = np.var(realizations)
print("Mean: {mean}\nDeviation: {std}\nVariance: {var}".format(mean=mu, std=std, var=variance))

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

plt.show()