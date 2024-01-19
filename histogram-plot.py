'''
Plot the histogram.
'''

import matplotlib.pyplot as plt 


# copy the ACF or PACF results from the exam into here
as_string = "0.43 0.26 0.14 0.08 -0.09 -0.07 -0.21 -0.11 -0.05 -0.01"
# or write them manually
history = []
# if you want to save the picture
save_plot = False

assert((as_string != "") or (len(history) > 0))

if as_string != "":
    history = [float(token) for token in as_string.split()]

if len(history) > 0:
    plt.stem(history)  
    if save_plot:
        plt.savefig("histogram.png")
    plt.show()
