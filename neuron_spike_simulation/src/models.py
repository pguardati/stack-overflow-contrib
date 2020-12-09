import math
import os
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime as dt

from neuron_spike_simulation.src.constants import LOG_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(
    os.path.join(LOG_DIR, dt.now().strftime('%Y-%m-%d--%H-%M-%S') + '_value.log')
)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# HH parameters
v_Rest = -65  # in mV
gNa = 1200  # in mS/cm^2
gK = 360  # in mS/cm^2
gL = 0.3 * 10  # in mS/cm^2
vNa = 115  # in mV
vK = -12  # in mV
vL = 10.6  # in mV
c = 1  # in uF/cm^2


# Introduction of equations and channels
def alphaM(v):
    return 12 * ((2.5 - 0.1 * (v)) / (np.exp(2.5 - 0.1 * (v)) - 1))


def betaM(v):
    return 12 * (4 * np.exp(-(v) / 18))


def betaH(v):
    return 12 * (1 / (np.exp(3 - 0.1 * (v)) + 1))


def alphaH(v):
    return 12 * (0.07 * np.exp(-(v) / 20))


def alphaN(v):
    return 12 * ((1 - 0.1 * (v)) / (10 * (np.exp(1 - 0.1 * (v)) - 1)))


def betaN(v):
    return 12 * (0.125 * np.exp(-(v) / 80))


def simulate_spike_dynamics(
        area_factor,
        stimulus,
        delay=0.1,
        dt=0.0025,
        duration=0.1,
        t_end=10
):
    """Step response of a neuron excited by an electrical signal
    Args:
        area_factor(float): multiplicative factor that determines the surface of the simulated neuron
        stimulus(float): amplitude of the input signal (volt)
        delay(float): delay of the input signal (s)
        dt(float): integration step (s)
        duration(float): total duration of the input (s)
        t_end(float): end of the simulation (s)

    Returns:

    """
    A = (1 * 10 ** (-8)) * area_factor  # surface [cm^2]
    C = c * A  # uF

    # compute the timesteps
    t_steps = t_end / dt + 1
    # Compute the initial values
    v0 = 0
    m0 = alphaM(v0) / (alphaM(v0) + betaM(v0))
    h0 = alphaH(v0) / (alphaH(v0) + betaH(v0))
    n0 = alphaN(v0) / (alphaN(v0) + betaN(v0))
    # Allocate memory for v, m, h, n
    v = np.zeros((int(t_steps), 1))
    m = np.zeros((int(t_steps), 1))
    h = np.zeros((int(t_steps), 1))
    n = np.zeros((int(t_steps), 1))
    # Set Initial values
    v[:, 0] = v0
    m[:, 0] = m0
    h[:, 0] = h0
    n[:, 0] = n0
    # Noise component
    knoise = 0.00005  # uA/(mS)^1/2

    for i in range(0, int(t_steps) - 1, 1):
        # Get current states
        vT = v[i]
        mT = m[i]
        hT = h[i]
        nT = n[i]

        # Stimulus current
        if delay / dt <= i <= (delay + duration) / dt:
            IStim = stimulus  # in uA
        else:
            IStim = 0

            #  Compute change of m, h and n
            m[i + 1] = (mT + dt * alphaM(vT)) / (1 + dt * (alphaM(vT) + betaM(vT)))
            h[i + 1] = (hT + dt * alphaH(vT)) / (1 + dt * (alphaH(vT) + betaH(vT)))
            n[i + 1] = (nT + dt * alphaN(vT)) / (1 + dt * (alphaN(vT) + betaN(vT)))

        # Ionic currents
        iNa = gNa * m[i + 1] ** 3. * h[i + 1] * (vT - vNa)
        iK = gK * n[i + 1] ** 4. * (vT - vK)
        iL = gL * (vT - vL)
        Inoise = (np.random.normal(0, 1) * knoise * np.sqrt(gNa * A))
        IIon = ((iNa + iK + iL) * A) + Inoise  #

        # Compute change of voltage
        v[i + 1] = (vT + ((-IIon + IStim) / C) * dt)[0]  # in ((uA / cm ^ 2) / (uF / cm ^ 2)) * ms == mV

        # stop simulation if it diverges
        if math.isnan(v[i + 1]):
            return [None]

    # adjust the voltage to the resting potential
    v = v + v_Rest
    return v


def HH_model(area_factor, stimulus, runs=1000, return_seq=False, t_end=10):
    """Predict how many simulations predicted at least one spike
    Args:
        area_factor(float): multiplicative factor that determines the surface of the simulated neuron
        stimulus(float): amplitude of the input signal (volt)
        runs(int): number of simulations
        return_seq(bool): if True, return the simulations that have been used to compute the spikes
        t_end(float): end of the simulation (s)

    Returns:
        int|tuple=[int,np.array]
    """
    count = 0
    delay = 0.1  # in ms
    duration = 0.1  # in ms
    dt = 0.0025  # in ms
    area_factor = area_factor

    simulations = []
    for _ in tqdm(range(0, runs), total=runs):
        v = simulate_spike_dynamics(area_factor, stimulus, delay, dt, duration, t_end)
        simulations.append(v)
        if max(v[:] - v_Rest) > 60:
            count += 1

    if return_seq:
        return count, simulations
    return count


def hh_model_normalised(percentage_of_stimulus, delta_v, delta_area, runs=10, t_end=2, v_ref=1e-7, area_ref=1):
    """HH model with normalised input and parameters
    Args:
        percentage_of_stimulus(float): percentage of the reference stimulus given to the neuron
        delta_v(float): multiplicative displacement of the stimulus voltage
        delta_area(float): multiplicative displacement of the contact area
        runs(int): number of times each simulation is executed
        t_end(int): end time of the simulation (s)
        v_ref(float): amplitude of the stimulus used to excite the neuron (V)
        area_ref(float): surface area of the neuron (cm^2)

    Returns:
        float: percentage of neuron spikes respect to the total runs
    """
    stimulus = v_ref * (1 + delta_v) * percentage_of_stimulus
    area_factor = area_ref * (1 + delta_area)
    y_counts = HH_model(area_factor, stimulus, runs=runs, t_end=t_end)
    normalized_count = y_counts / runs
    return normalized_count


def linear_model(x, delta_v, delta_area, *args):
    """linear predictor, used for test purposes"""
    y = x * delta_v + delta_area
    return y


def mse_loss(parameters, model, x_ref, y_ref, runs=10, t_end=2, return_prediction=False):
    """compute mean square error between the prediction of a model and the input data"""
    delta_v, delta_area = parameters
    y_predicted = np.array([model(x, delta_v, delta_area, runs, t_end) for x in x_ref])
    mse = ((y_ref - y_predicted) ** 2).mean()
    logger.info("delta_v:{:.5e}, delta_area:{:.5e}, mse:{:.5f}".format(
        delta_v,
        delta_area,
        mse
    ))
    if return_prediction:
        return mse, y_predicted
    return mse
