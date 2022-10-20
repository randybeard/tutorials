"""
messing with b-splines
        2/18/21 - RWB
        10/19/22 - RWB
"""
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def splineBasis(order, knots, controlPoints, time):
        spl = BSpline(knots, controlPoints, order)
        basisPoints = spl(time)
        return( basisPoints )

def plotSplineBasis(order):
        fig = plt.figure(order)
        if order > 0:
                #N = order + 1
                N = order + 1 + 1
        else:
                N = 2
        t = np.linspace(0, N, 100)
        knots = [0] * order + list(range(0, N+1)) + [N] * order
        fig.suptitle(f"Order={order}, knots = {str(knots)}")
        ax = fig.subplots(N+order)
        for i in range(0, N+order):
                ctrl = [0] * (N+order)
                ctrl[i] = 1
                pts = splineBasis(order, knots, ctrl, t)
                ax[i].plot(t, pts)
                ax[i].set(ylabel=f"m={i}")
 
plotSplineBasis(1)
plt.show()


