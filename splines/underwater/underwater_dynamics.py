from scipy.integrate import odeint
import numpy as np
import random
import simulation_parameters as P


class underwater_vehicle:
    """
    underwater vehicle
    """

    def __init__(self):
        # states are forward position p and forward velocity v
        self.state_ = [P.x0]  # history of states
        # input is thrust
        self.input_ = [0]  # history of inputs
        # output is position
        self.output_ = [0]  # history of outputs
        # parameters of the system
        # the parameters will be different with each simulation to test
        # the robustness of the controllers
        beta = 0.2  # level of uncertainty in the control parameters
        self.m = P.m*random.uniform(1-beta, 1+beta)   # kg
        self.mu1 = P.mu1*random.uniform(1-beta, 1+beta)   # kg / m
        self.mu2 = P.mu2*random.uniform(1-beta, 1+beta)   # kg s^2 / m^3
        self.alpha = P.alpha*random.uniform(1-beta, 1+beta)   # unitless
        self.vc = P.vc*random.uniform(1-beta, 1+beta)   # m / 2
        self.vc = P.vc # speed of the current
        self.Tmax = P.Tmax  # maximum allowable thrust
        self.Ts = P.Ts  # sample rate

    def dynamics(self, x, t, u):
        """Define dynamics xdot=f(x,t,u). Return xdot."""
        # saturate inpute
        u = self.sat(u, self.Tmax)
        xdot = [0.0, 0.0]
        v = x[1]
        # calculate dynamics
        xdot[0] = v + self.vc
        xdot[1] = -self.mu1/self.m*v*abs(v) - self.mu2/self.m*abs(v)*(v**3) + \
            self.alpha/self.m*u
        # return derivatives
        return xdot

    def integrate(self, u, t1, t2):
        """Integrate dynamics from t1 to t2 with N time steps."""
        N = 5  # number of time instances between t1 and t2
        x_next = odeint(self.dynamics, self.state_[-1],
                        np.arange(t1, t2, (t2-t1)/N), args=(u,))
        self.state_ = np.concatenate((self.state_,
                                     np.vsplit(x_next, x_next.shape[0])[-1]))
        self.input_ = np.append(self.input_, u)
        self.output_ = np.append(self.output_, self.output())

    def state(self):
        """Return current state."""
        return self.state_[-1]

    def output(self):
        """Return current output."""
        x = self.state()
        y = x[0]
        return y

    def getStateHistory(self):
        """Return state history."""
        return self.state_

    def getInputHistory(self):
        """Return input history."""
        return self.input_

    def getOutputHistory(self):
        """Return output history."""
        return self.output_

    def sat(self, u, limit=1):
        """Saturates u at the +-limit"""
        if u >= limit:
            y = limit
        elif u <= -limit:
            y = -limit
        else:
            y = u
        return y
