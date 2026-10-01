# Written by Luke Smith
# Started 9/29
# Goal of this program is to simulate several time steps of a particle using Langevin Overdamped Dynamics
# and the muller-brown potential energy surface

import math
import matplotlib.pyplot as plt
import numpy as np
import random


# Standard Parameters
n = 4
A = [-200, -100, -170, 15]
a = [-1, -1, -6.5, 0.7]
b = [0, 0, 11, 0.6]
c = [-10, -10, -6.5, 0.7]
x0 = [1, 0, -0.5, -1]
y0 = [0, 0.5, 1.5, 1]


def gradx(xval, yval):
    # Calculate the gradiant of V with respect to x, using the Muller-Brown Potential Energy surface
    # found on this website: https://hunterheidenreich.com/notes/chemistry/molecular-simulation/classical-methods/muller-brown-1979/
    dVdx = 0
    # numpy array, numpy.sum
    # we <3 vectorization
    for k in range(0, n):
        gkxy = a[k]*(xval - x0[k])**2 + b[k]*(xval-x0[k])*(yval - y0[k]) + c[k]*(yval - y0[k])**2
        try:
            try_exp = math.exp(gkxy)
        except OverflowError:
            print("Solution diverged for given IC and dt. Try a smaller dt or a different IC")
            raise
        dVdx += A[k]*math.exp(gkxy)*(2*a[k]*(xval - x0[k])+b[k]*(yval-y0[k]))
    return dVdx


def grady(xval, yval):
    # Calculate the gradiant of V with respect to y, using the Muller-Brown Potential Energy surface
    # found on this website: https://hunterheidenreich.com/notes/chemistry/molecular-simulation/classical-methods/muller-brown-1979/
    dVdy = 0
    for k in range(0, n):
        gkxy = a[k]*(xval - x0[k])**2 + b[k]*(xval-x0[k])*(yval - y0[k]) + c[k]*(yval - y0[k])**2
        try:
            try_exp = math.exp(gkxy)
        except OverflowError:
            print("Solution diverged for given IC and dt. Try a smaller dt or a different IC")
            raise
        dVdy += A[k]*math.exp(gkxy)*(b[k]*(xval - x0[k])+2*c[k]*(yval-y0[k]))
    return dVdy


if __name__ == "__main__":
    # set up soln results
    xvals = []  # switch to numpy array
    yvals = []
    # Initial Conditions

    # Saddle point:
    # xnaught = -0.8
    # ynaught = 0.6

    # Two minima:
    # xnaught = 0.5
    # ynaught = 0.4

    # big minima:
    # xnaught = -0.6
    # ynaught = 1.4

    # far away:
    # xnaught = -1.5
    # ynaught = 0.5

    xnaught = -0.5
    ynaught = 1

    xvals.append(xnaught)
    yvals.append(ynaught)

    # set up variables
    dt = 1*10**(-5)  # time step size, 1e-4, 1e-3, smaller for bigger
    # use an array of size numsteps, index for t should match
    numsteps = 10**6
    tend = dt*numsteps
    tvals = np.arange(0, tend, dt)

    D = 2  # Diffusivity
    Kbeta = 1  # Boltzman constant
    T = 1  # temperature
    gamma = 1  # damping
    m = 1  # mass
    for t in range(0, numsteps-1):
        # check if this is the right equation for dx, dy
        dxcurr = -gradx(xvals[t], yvals[t])*dt + math.sqrt(2*m*D*Kbeta*T*dt)*random.gauss(mu=0.0, sigma=1.0)
        dycurr = -grady(xvals[t], yvals[t])*dt + math.sqrt(2*m*D*Kbeta*T*dt)*random.gauss(mu=0.0, sigma=1.0)
        xvals.append(xvals[t] + dxcurr)
        yvals.append(yvals[t] + dycurr)
    plt.plot(xvals, yvals)
    plt.xlabel("X position over time")
    plt.ylabel("Y position over time")
    plt.title(f"Position over time, dt = {dt} and num steps = {numsteps}")
    minimaX = [-0.558, -0.050, 0.623]
    minimaY = [1.442, 0.467, 0.028]
    saddleX = [-0.822, 0.212]
    saddleY = [0.624, 0.293]
    plt.plot(xnaught, ynaught, 'bo', label="Starting Point")
    plt.plot(minimaX, minimaY, 'c+', label="Local Minima")
    plt.plot(saddleX, saddleY, 'rs', label="Saddle Points")
    plt.legend()
    plt.show()
