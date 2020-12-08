import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from neuron_spike_simulation.src.models import hh_model_normalised, mse_loss

# data to validate the model
x_data = np.array([0.92, 0.925, 0.9535, 0.975, 0.9789, 1, 1.02, 1.043, 1.06, 1.078, 1.09])
y_data = np.array([150, 170, 269, 360, 377, 500, 583, 690, 761, 827, 840]) / 1000


def run(args):
    print("Running {} spikes simulation for {} input stimuli".format(args.runs, len(x_data)))
    parameters = args.delta_v, args.delta_area
    mse, y_predicted = mse_loss(
        parameters,
        hh_model_normalised,
        x_data,
        y_data,
        args.runs,
        args.t_end,
        return_prediction=True
    )
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('input')
    ax.set_ylabel('output predictions')
    ax.plot(x_data, y_data, marker='o', markersize=10)
    ax.plot(x_data, y_predicted, marker='*')
    ax.legend(["raw data points", "initial guess", "predictions with optimized parameters"])
    plt.show()


def parse_args(args=None):
    parser = argparse.ArgumentParser("Predict number of spikes and plot model results")
    parser.add_argument("--delta-v", default=0, type=float,
                        help="normalised displacement respect to maximum applicable voltage")
    parser.add_argument("--delta-area", default=0, type=float,
                        help="surface of contact between the input electrode and the neuron")
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
    main(['--delta-v=0', '--delta-area=0', '--runs=10'])
