# controllers for underwater vehicle
import numpy as np
import simulation_parameters as P

class controller:
    def __init__(self, integrator0=0.0):
        # -- memory variables for integrator (PID and LQR) --
        self.integrator = integrator0  # initial condition for integrator
        self.error_d1 = 0.0  # delay of input to integrator

        # -- memory variables for dirty derivative --
        self.y_d1 = 0.0  # delay of y for differentiator
        self.ydot = 0.0  # output of differentiator
        # gains for dirty derivative
        self.diff_gain1 = (2.0*P.sigma-P.Ts)/(2.0*P.sigma+P.Ts)
        self.diff_gain2 = 2.0/(2.0*P.sigma+P.Ts)
        
        # -- states for adaptive controller --
        self.xref = np.matrix(np.zeros((P.Gam.shape[0],1)))
        self.thetahat = np.matrix(np.zeros((P.Gam.shape[0],1)))

    def PID(self, x, r):
        # read in states        
        y = x[0]  # position
        v = x[1]  # velocity
        # compute the error
        error = y-r
        # integrate the error
        if np.abs(v) < 0.5:  # crude anti-windup
            self.integrator = self.integrator \
                                + (P.Ts/2.0)*(error+self.error_d1)
        self.error_d1 = error
        # PID controller before saturation
        u_unsat = - P.pid_kp*error \
                  - P.pid_ki*self.integrator \
                  - P.pid_kd*v
        u = self.sat(u_unsat, P.Tmax)
        if P.pid_ki != 0.0:
            self.integrator = self.integrator + (P.Ts/P.pid_ki)*(u-u_unsat)
        return u

    def backstepping(self, x, r):
        # read in full states        
        y = x[0]  # position
        v = x[1]  # velocity
        # compute the error
        error = y-r
        # backstepping controller
        zeta = -P.vc - P.backstepping_k1*error
        zetadot = -P.backstepping_k1*(v+P.vc)
        T = P.m/P.alpha*( -error + P.mu1/P.m*v*np.abs(v) \
                         + P.mu2/P.m*v*np.abs(v)**3 \
                         + zetadot \
                         - P.backstepping_k2*(v-zeta) \
                         )
        return self.sat(T, P.Tmax)

    def feedback_linearization(self, x, r):
        # read in full states        
        y = x[0]  # position
        v = x[1]  # velocity
        # feedback linearizing control
        x1 = y-r
        x2 = v + P.vc
        nu = -P.flc_k1*x1 - P.flc_k2*x2
        T = P.m/P.alpha*( P.mu1/P.m*v*np.abs(v) \
                         + P.mu2/P.m*v*np.abs(v)**3 \
                         + nu \
                         )
        return self.sat(T, P.Tmax)

    def sliding_mode(self, x, r):
        # read in full states        
        y = x[0]  # position
        v = x[1]  # velocity
        # compute the error
        error = y-r
        # sliding mode controller
        s = v + P.vc + P.sm_ks*error;
        rho = (1/P.sm_alpha_min)*(P.sm_mu1_max*v**2 \
              + P.sm_mu2_max*v**4 \
              + P.sm_ks*P.sm_m_max*(P.sm_vc_max + abs(v)))
        T = -(rho+P.sm_beta0)*self.sat(s/P.sm_epsilon, 1.0)
        return T

    def lqr(self, x, r):
        # read in full states        
        y = x[0]  # position
        v = x[1]  # velocity
        # compute the error
        error = y-r
        # integrate the error
        self.integrator = self.integrator + (P.Ts/2.0)*(error+self.error_d1)
        self.error_d1 = error
        # lqr control
        augmented_state = np.matrix([
                                   [self.integrator],
                                   [y],
                                   [v]
                                   ])
        T_unsat = - float(P.K_lqr*augmented_state)
        T = self.sat(T_unsat, P.Tmax)
        if P.K_lqr[0][0] != 0.0:
            self.integrator = self.integrator + (P.Ts/P.K_lqr[0][0])*(T-T_unsat)
        return T, augmented_state

    def adaptive(self, x, r):
        # position = x[0]
        v = x[1]     
        # baseline LQR+integral controller
        u_lqr, augmented_state = self.lqr(x, r)
        # build regressor vector
        Phi = np.matrix([  
                [u_lqr],  # baseline controller
                [v*np.abs(v)], 
                [v*np.abs(v)**3],
                ])
        # update reference model and adaptive parameters
        N = 10
        for i in range(0,N):
            self.xref = self.xref + P.Ts/N*( \
                        P.Aref*self.xref + P.Bref*r \
                        + P.L_nu*(augmented_state-self.xref))
            # p_ref = P.Cref*self.xref
            e = self.deadzone(augmented_state-self.xref, P.deadzone_limit)
            self.thetahat = self.thetahat + P.Ts/N*self.proj(self.thetahat, 
                              P.Gam*Phi*np.transpose(e)*P.P_nu_inverse*P.B)
            for j in range(0, self.thetahat.shape[0]):
                self.thetahat[j] = self.sat(self.thetahat[j], P.proj_limit[j])
        u_adaptive = -np.transpose(self.thetahat)*Phi;    
        T = u_lqr + u_adaptive;        
        return self.sat(T, P.Tmax)

    def sat(self, u, limit=1):
        """Saturates u at the +-limit"""
        if u >= limit:
            y = limit
        elif u <= -limit:
            y = -limit
        else:
            y = u
        return y

    def deadzone(self, y, limit):
        """Deadzone: zeros elements close to zero"""
        y_deadzone = y
        N = len(limit)
        for i in range(0,N):
            if np.abs(y[i]) <= limit[i]:
                y_deadzone[i]=0
        return y_deadzone
        
    def proj(self, theta, y):
        """Projection operator for adaptive control"""
        yout = y;
        delta = 0.1;
        N = theta.shape[0]
        for i in range(0, N):
            if theta[i]>(P.proj_limit[i]-delta) and y[i]>0:
                yout[i] = ((P.proj_limit[i]-theta[i])/delta)*y[i]
            if theta[i]<(-P.proj_limit[i]+delta) and y[i]<0:
                yout[i] = ((theta[i]-P.proj_limit[i])/delta)*y[i] 
        return yout

class observer:
    def __init__(self):
        # -- memory variables for dirty derivative --
        self.y_d1 = 0.0  # delay of y for differentiator
        self.ydot = 0.0  # output of differentiator
        # gains for dirty derivative
        self.diff_gain1 = (2.0*P.sigma-P.Ts)/(2.0*P.sigma+P.Ts)
        self.diff_gain2 = 2.0/(2.0*P.sigma+P.Ts)

    def dirty_differentiator(self, y):
        # dirty derivative of y to get velocity
        self.ydot = self.diff_gain1*self.ydot+self.diff_gain2*(y-self.y_d1)
        self.y_d1 = y
        # construct estimate of state
        xhat = [y, self.ydot]
        return xhat
