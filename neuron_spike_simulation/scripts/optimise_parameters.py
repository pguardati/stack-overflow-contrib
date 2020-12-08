import sys
import argparse
import numpy as np
from neuron_spike_simulation.src.models import hh_model_normalised, mse_loss
from scipy.optimize import minimize

x_data = np.array([0.92, 0.925, 0.9535, 0.975, 0.9789, 1, 1.02, 1.043, 1.06, 1.078, 1.09])
y_data = np.array([150, 170, 269, 360, 377, 500, 583, 690, 761, 827, 840]) / 1000


def run(args):
    print("Running {} spikes simulation for {} input stimuli".format(args.runs, len(x_data)))
    optimum = minimize(
        mse_loss,
        x0=np.array([0, 0]),
        args=(hh_model_normalised, x_data, y_data, args.runs, args.t_end),
        options={"disp": True},
        bounds=((-0.9, 9), (-0.9, 9))
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser("Predict number of spikes and plot model results")
    parser.add_argument("--runs", default=100, type=int,
                        help="number of time each simulation is run to predict one output")
    parser.add_argument("--t-end", default=2, type=float, help="time when the simulation ends (ms)")
    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    run(args)


if __name__ == "__main__":
    main(['--runs=10'])
