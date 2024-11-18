""" 
This collection of files is based on daniel-s-ingram's visualizations 
of the course "Artificial Intelligence for Robotics" by Sebastian Thrun

This particular file focuses on probablistic 2D localization.
"""

# Imports
from __future__ import print_function, division
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#
dt = 0.1
I = np.identity(4)                        # Identity Matrix
F = np.array([[1, 0, dt,  0],             # State Transition Matrix
              [0, 1,  0, dt],
              [0, 0,  1,  0],
              [0, 0,  0,  0]])
H = np.array([[1, 0, 0, 0],               # Measurement Function
              [0, 1, 0, 0]])
R = np.array([[100,   0],                 # Measurement Noise
              [  0, 100]])
x = np.array([[50],                       # State Estimation
              [50],
              [ 0],
              [ 0]])
P = np.array([[1000,    0,    0,    0],   # Uncertainty Covariance
              [   0, 1000,    0,    0],
              [   0,    0, 1000,    0],
              [   0,    0,    0, 1000]])
u = np.array([[0],                        # Control
              [0],
              [0],
              [0]])

def predict(x, u, P):
    x = F.dot(x) + u
    P = F.dot(P).dot(F.T)
    return x, P

def update(x, z, P):
    y = z.T - H.dot(x)
    S = H.dot(P).dot(H.T) + R
    K = P.dot(H.T).dot(np.linalg.inv(S))
    x = x + K.dot(y)
    P = (I - K.dot(H)).dot(P)
    return x, P

def gaussian_2d(X, Y, x, P):
    return np.exp(-((X - x[0, 0])**2/(2*P[0, 0]) + (Y - x[1, 0])**2/(2*P[1, 1])))

#
fig, ax = plt.subplots()
X, Y = np.meshgrid([1.0*i for i in range(100)], [1.0*i for i in range(100)])
action_space = [[ 1,  0,  0,  0],
                [ 0,  1,  0,  0],
                [-1,  0,  0,  0],
                [ 0, -1,  0,  0],
                [ 1,  1,  0,  0],
                [ 1, -1,  0,  0],
                [-1,  1,  0,  0],
                [-1, -1,  0,  0]]
motions = np.array([random.choice(action_space) for _ in range(500)])
measurements = np.array([[[50, 50]] for _ in range(500)])
for i in range(1, 500):
    measurements[i, :] = measurements[i-1, :] + motions[i, :2]

f = gaussian_2d(X, Y, x, P)
grid = ax.imshow(f, cmap="GnBu")
dot, = ax.plot(0, 0, 'r.')

def animate(i):
    global x, P
    x, P = predict(x, motions[i].reshape(4, 1), P)
    x, P = update(x, measurements[i].reshape(1, 2), P)
    f = gaussian_2d(X, Y, x, P)
    grid.set_data(f)
    smallest = min(min(row) for row in f)
    largest = max(max(row) for row in f)
    grid.set_clim(vmin=0, vmax=1)
    dot.set_data(measurements[i, 0, :2])
    return grid, dot,

def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    return grid, dot,

anim = animation.FuncAnimation(fig, animate, 500, interval=50, init_func=init)
plt.show()