import os
import math
import sys
import itertools
import argparse
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import rc

from neuron_spike_simulation.src.models import HH_model
from neuron_spike_simulation.src.constants import LOG_DIR

rc('text', usetex=True)


def plot_spikes_vs_surfaces_and_stimuli(df):
    """Plot spikes as function of the input stimuli and neuron surfaces
    Args:
        df(pd.DataFrame): dataset that contains
                            -'area': surface of a neuron
                            -'stimulus': input voltage
                            -'spikes': number of spikes associated to a given area and stimulus

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(r"dB of Stimulus (V)")
    ax.set_ylabel(r"Number of Spikes")
    # scale stimulus to decibel and plot
    df["stimulus_db"] = df["stimulus"].apply(lambda x: 10 * math.log10(x))
    for key, grp in df.groupby(['area']):
        ax.plot(grp['stimulus_db'], grp['spikes'], label=key)
    ax.legend(loc='upper right', title='Surface of a Neuron')
    return fig


def run(args):
    print("Create grid of parameters..")
    voltage_interval = np.logspace(
        start=args.voltage_exp_start,
        stop=args.voltage_exp_end,
        num=args.voltage_points,
        base=10
    )
    area_interval = np.logspace(
        start=args.area_exp_start,
        stop=args.area_exp_end,
        num=args.area_points,
        base=10
    )
    param_combination = list(itertools.product(*[area_interval, voltage_interval]))

    print("Evaluate simulation on {} combinations of input stimulus and surface area..".format(len(param_combination)))
    simulations = []
    for i, (area_factor, stimulus) in enumerate(param_combination):
        try:
            res = HH_model(area_factor, stimulus, return_seq=True, runs=args.runs, t_end=args.t_end)
        except:
            print("simulation {} failed for area:{} and stimulus:{}".format(i, area_factor, stimulus))
            res = None
        simulations.append(res)

    print("Create simulation datasets..")
    df = pd.DataFrame(param_combination, columns=["area", "stimulus"])
    # dataset to sum up simulation results
    df_summary = df.copy()
    df_summary["spikes"] = list(zip(*simulations))[0]
    # dataset to store the full simulations
    df_sim = df_summary.copy()
    sequence = list(zip(*simulations))[1]
    sequence_flatten = [np.stack(seq, axis=0).squeeze() for seq in sequence]
    df_sim["sequence"] = sequence_flatten

    if args.store_results:
        path_summary = os.path.join(LOG_DIR, dt.now().strftime('%Y-%m-%d--%H-%M-%S') + 'df_summary.csv')
        path_sim = os.path.join(LOG_DIR, dt.now().strftime('%Y-%m-%d--%H-%M-%S') + 'df_sim.csv')
        os.makedirs(LOG_DIR, exist_ok=True)
        print("Store summary in {} and full simulation in {}".format(path_summary, path_sim))
        df_summary.to_csv(path_summary, index=False)
        df_sim.to_csv(path_sim, index=False)

    if args.store_plots:
        fig = plot_spikes_vs_surfaces_and_stimuli(df_summary)
        path_fig = os.path.join(LOG_DIR, dt.now().strftime('%Y-%m-%d--%H-%M-%S') + 'plot.png')
        print("Store plot in {}".format(path_fig))
        fig.savefig(path_fig)


def parse_ars(args):
    parser = argparse.ArgumentParser(
        description="Run spike simulations for a grid of values of voltage stimulus and surface of a neuron")
    parser.add_argument("--voltage-points", default=10, type=int, help="number of voltage values to explore")
    parser.add_argument("--voltage-exp-start", default=-8, type=float,
                        help="order of magnitude of the minimum value of the voltage grid")
    parser.add_argument("--voltage-exp-end", default=-7, type=float,
                        help="order of magnitude of the maximum value of the voltage grid")

    parser.add_argument("--area-points", default=1, type=int, help="number of area values to explore")
    parser.add_argument("--area-exp-start", default=-1, type=float,
                        help="order of magnitude of the minimum value of the area grid")
    parser.add_argument("--area-exp-end", default=-1, type=float,
                        help="order of magnitude of the maximum value of the area grid")

    parser.add_argument("--runs", default=10, type=int,
                        help="number of time each simulation is run to predict one output")
    parser.add_argument("--t-end", default=2, type=float, help="time when the simulation ends (ms)")
    parser.add_argument("--store-results", action="store_true", help="store simulation results in a csv")
    parser.add_argument("--store-plots", action="store_true", help="store result plot when the simulation stops")
    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_ars(args)
    run(args)


if __name__ == "__main__":
    main([
        "--voltage-points=100",
        "--voltage-exp-star=-7.5",
        "--voltage-exp-end=-6.5",
        "--area-points=1",
        "--area-exp-start=-1",
        "--area-exp-end=-1",
        "--store-results",
        "--store-plots"
    ])
