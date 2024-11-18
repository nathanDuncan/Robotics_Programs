""" 
This collection of files is based on daniel-s-ingram's visualizations 
of the course "Artificial Intelligence for Robotics" by Sebastian Thrun

This particular file focuses on probablistic 1D localization.
"""

# Imports
from __future__ import print_function, division
from random import gauss
from math import sqrt, exp, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def f(mean, var, x):
    """
    Guassian probability distribution function

    Inputs
    ------
    mean : float
        mean of distribution
    var : float
        varianace of distribution
    x : float
        query point (independent variable)
    
    Returns
    -------
        Probability of x
    """
    return exp(-0.5*(x-mean)**2/var)/sqrt(2*pi*var) 

def update(mean1, var1, mean2, var2):
    """
    Updates belief distribution (1) based on a new measurement (2)
    
    Inputs
    ------
    mean1 : float
        mean of prior distribution
    var1 : float
        variance of prior distribution
    mean2 : float
        mean of observation distribution
    var2 : float
        variance of observation distribution
        
    Returns
    -------
        Updated belief distribution parameters (mean, variance)
    """

    new_mean = (var2*mean1 + var1*mean2)/(var1 + var2)
    new_var = 1/(1/var1 + 1/var2)
    return new_mean, new_var

def predict(mean1, var1, mean2, var2):
    """
    Updates belief distribution (1) based on a new movement (2)
    
    Inputs
    ------
    mean1 : float
        mean of prior distribution
    var1 : float
        variance of prior distribution
    mean2 : float
        mean of movement distribution (should be the desired movement)
    var2 : float
        variance of movement distribution
        
    Returns
    -------
        Updated belief distribution parameters (mean, variance)
    """

    new_mean = mean1 + mean2
    new_var = var1 + var2
    return new_mean, new_var

n_steps = 50
measurement_var = 0.5
motion_var = 0.5

# Measurements aren't really based on anything
measurements = [gauss(i, measurement_var) for i in range(n_steps)] 
motion = [gauss(1, motion_var) for _ in range(n_steps)]

# Initial belief
mean = 0
var = 10000

# linespace for plotting
x_pts = [i/10 for i in range(500)]

#
fig, ax = plt.subplots()

y1, = ax.plot(x_pts, x_pts)
y2, = ax.plot(x_pts, x_pts)
y3, = ax.plot(x_pts, x_pts)

def animate(i):
    global mean, var
    mean, var = predict(mean, var, motion[i], motion_var)
    y1_pts = [f(mean, var, x) for x in x_pts]
    mean, var = update(mean, var, measurements[i], measurement_var)
    y2_pts = [f(measurements[i], measurement_var, x) for x in x_pts]
    y3_pts = [f(mean, var, x) for x in x_pts]

    y1.set_ydata(y1_pts)
    y2.set_ydata(y2_pts)
    y3.set_ydata(y3_pts)
    
    return y1, y2, y3,

def init(): 
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, 1)
    ax.legend(("Prediction", "Measurement", "Update"))
    return y1, y2, y3,

anim = animation.FuncAnimation(fig, animate, 50, interval=100, init_func=init)
plt.show()