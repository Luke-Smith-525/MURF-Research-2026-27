# Written by Luke Smith
# Started 9/29
# Goal of this program is to simulate several time steps of the Muller Brown function

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
    dVdx = 0
    for k in range(0, n):
        gkxy = a[k]*(xval - x0[k])**2 + b[k]*(xval-x0[k])*(yval - y0[k]) + c[k]*(yval - y0[k])**2
        print(gkxy)
        dVdx += A[k]*math.exp(gkxy)*(2*a[k]*(xval - x0[k])+b[k]*(yval-y0[k]))
    return dVdx


def grady(xval, yval):
    dVdy = 0
    for k in range(0, n):
        gkxy = a[k]*(xval - x0[k])**2 + b[k]*(xval-x0[k])*(yval - y0[k]) + c[k]*(yval - y0[k])**2
        dVdy += A[k]*math.exp(gkxy)*(b[k]*(xval - x0[k])+2*c[k]*(yval-y0[k]))
    return dVdy


if __name__ == "__main__":
    # set up soln results
    xvals = []
    yvals = []
    xnaught = 0
    ynaught = 0
    xvals.append(xnaught)
    yvals.append(ynaught)

    # set up variables
    tend = 10
    dt = 0.1
    tvals = np.arange(0, tend, dt)
    D = 2  # Diffusivity
    Kbeta = 1  # Boltzman constant
    T = 1  # temperature
    gamma = 1  # damping
    m = 1  # mass
    for t in range(0, len(tvals)-1):
        # check if this is the right equation for dx, dy
        dxcurr = -gradx(xvals[t], yvals[t])*dt + math.sqrt(2*m*D*Kbeta*T*dt)*random.gauss(mu=0.0, sigma=1.0)
        dycurr = -grady(xvals[t], yvals[t])*dt + math.sqrt(2*m*D*Kbeta*T*dt)*random.gauss(mu=0.0, sigma=1.0)
        xvals.append(xvals[t] + dxcurr)
        yvals.append(yvals[t] + dycurr)
    plt.plot(xvals, yvals)
    plt.show()