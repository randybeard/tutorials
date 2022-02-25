# simulation parameters for underwater vehicle
import numpy as np
import control

# initial conditions
x0 = [
        0,  # initial position in meters
        0,  # initial velocity in meters/second
      ]

# physical parameters known to controller
m = 10.0  # kg mass
mu1 = 25.0  # kg / m
mu2 = 15.0  # kg s^2 / m^3
alpha = 0.8  # unitless
Tmax = 50.0  # maximum allowable thrust
vc = 0.5  # m / s

# sample rate
Ts = 0.01

# parameter for dirty derivative
sigma = 0.05

# -------------------------------------------------
# parameters for PID controller
pid_kp = 25.0
pid_kd = 20.0
pid_ki = 1.0

# -------------------------------------------------
# parameters for backstepping controller
backstepping_k1 = 1.0
backstepping_k2 = 1.0

# -------------------------------------------------
# parameters for feedback linearizing controller
flc_zeta = 0.707
flc_wn = 0.8
flc_k1 = flc_wn**2
flc_k2 = 2.0*flc_zeta*flc_wn

# -------------------------------------------------
# parameters for sliding mode controller
sm_ks = 2.0
sm_beta0 = 1.0
sm_epsilon = 0.1
sm_mu1_max = 2.0*mu1
sm_mu2_max = 2.0*mu2
sm_alpha_min = 1.0
sm_m_max = 2.0*m
sm_vc_max = 2.0*vc

# -------------------------------------------------
# parameters for LQR controller

# state space model based on Jacobian linearization
# states are x=[p, v]
Ap = np.matrix([[0, 1], [0, 0]])
Bp = np.matrix([[0], [1]])
Cp = np.matrix([[1, 0]])

# form augmented system to add integrator
Cout = np.matrix([[1, 0]])
zeros11 = np.matrix([[0]])
zeros21 = np.matrix([[0], [0]])
A = np.bmat([[zeros11, Cout], [zeros21, Ap]])
B = np.bmat([[zeros11], [Bp]])

# design LQR controller
Q = np.matrix(np.diag([10, 5, 200]))
R = 1.0/Tmax  # penalty on thrust
K_lqr, P_lqr, E_lqr = control.lqr(A, B, Q, R)

# -------------------------------------------------
# parameters for adaptive controller

# construct reference model
Aref = A-B*K_lqr
Bref = np.matrix([[-1], [0], [0]])
Cref = np.bmat([[np.matrix([[0]]), Cout]])

# observer in the reference model
nu = 0.001
L, S, E = control.lqr(np.transpose(Aref),
                      np.matrix(np.identity(3)),
                      (nu+1)/nu*np.matrix(np.identity(3)),
                      nu/(nu+1)*np.matrix(np.identity(3)))
L_nu = np.transpose(L)
P_nu_inverse = np.linalg.inv(S)

# adaptive gains
Gam = 10.0*np.matrix(np.diag(np.array([
                                     1.0,  # baseline controller
                                     1.0,  # v*abs(v)
                                     1.0,  # v*abs(v)^3
                                     ])))
num_adaptive_param = Gam.shape[0]
deadzone_limit = [
                0.0,  # limit on integral error
                0.0,  # limit on p
                0.0,  # limit on v
                ]
proj_limit = [
              2.0,  # baseline controller
              2.0,  # v*abs(v)
              2.0,  # v*abs(v)^3
              ]
