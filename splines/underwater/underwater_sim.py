import numpy as np
import matplotlib.pyplot as plt
import simulation_parameters as P
from underwater_dynamics import underwater_vehicle
from underwater_ctrl import controller
from underwater_ctrl import observer


def square_wave(t, A, f):
    """Return a square wave at time t with amplitude A and frquency f.
        t -- the current time
        A -- the amplitude of the square wave
        f -- the frequency of the square wave in Hz
    """
    if t % (1.0/f) <= (1.0/(2.0*f)):
        r = A
    else:
        r = -A
    return r


def plot_results(t, x, xhat, r, u):
    """Plot the results of the simulation.
        t -- time        
        x -- state
        xhat -- estimated state
        r -- reference
        u -- input
    """
    # set up window and define subplots    
    plt.clf()
    fig1, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True, num=1)
    # subplot 1,1
    ax1.plot(t, x[:, 0], label='y')  # plot x[0]
    ax1.plot(t, r, label='r')  # plot reference
    ax1.set_ylabel('position (m)')
    ax1.legend()  # add legend to first plot
    ax1.grid(True)  # adds grid to first plot
    # subplot 2,1
    ax2.plot(t, u, label='u')
    ax2.set_ylabel('Thrust (N)')
    ax2.legend()
    ax2.grid(True)
    #subplot 3,1
    for i in range(0, x.shape[1]):
        ax3.plot(t, xhat[:, i])
        ax3.plot(t, x[:, i], label='x[{:d}]'.format(i))
    ax3.set_ylabel('x and xhat (m/s)')
    ax3.legend()
    ax3.grid(True)
    # labels for window
    plt.xlabel('time (s)')
    plt.title('Simulation Results')
    plt.show()


# ----------------------------------------------------------------------------
# simulate the closed loop system
# ----------------------------------------------------------------------------
# instantiate underwater vehicle
plant = underwater_vehicle()
# instantiate observer
obsv = observer()
# instantiate controller
ctrl = controller()
# simulation start and stop times
sim_start_time = 0.0  
sim_stop_time = 200.0  
# reference signal (square wave)
reference_amplitude = 1.0  # meters
reference_frequency = 0.01  # Hertz
# simulation loop
t = sim_start_time
while t < sim_stop_time:
    # -- get the current output of the plant y(t) --
    y = plant.output()
    # -- estimate state using observer -- 
    # xhat = obsv.dirty_differentiator(y)  # use reconstructed state
    xhat = plant.state()  # use actual state
    # -- compute reference signal r(t) --
    r = square_wave(t, reference_amplitude, reference_frequency)
    # -- compute the control signal u(t) --
    # u = ctrl.PID(xhat, r)
    # u = ctrl.backstepping(xhat, r)
    # u = ctrl.feedback_linearization(xhat, r)
    # u = ctrl.sliding_mode(xhat, r)
    # u, foo = ctrl.lqr(xhat, r)
    u = ctrl.adaptive(xhat, r)
    # -- integrate dynamics for next Ts seconds with constant input u -- 
    plant.integrate(u, t, t+P.Ts)
    # -- create log of reference, estimated state, and time for plotting -- 
    if t == sim_start_time:
        reference = [r]
        x_hat = [xhat]
        time = [sim_start_time]
    reference = np.append(reference, r)
    x_hat = np.concatenate((x_hat, [xhat]), axis=0)
    time = np.append(time, t+P.Ts)
    # -- update simulation time -- 
    t += P.Ts

# plot the results
# %matplotlib qt <- type this in spyder for separate plot window
# %matplotlib inline <- type this for inline windows
plot_results(time, plant.getStateHistory(), x_hat, 
             reference, plant.getInputHistory())
