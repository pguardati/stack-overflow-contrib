import unittest
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from neuron_spike_simulation.src.models import simulate_spike_dynamics, HH_model, hh_model_normalised, linear_model, \
    mse_loss
from neuron_spike_simulation.scripts import run_grid_search_simulation

SHOW_PLOT = False


class TestEquations(unittest.TestCase):

    def test_simulation_convergence(self):
        """test the convergence of one simulation"""
        voltage_dynamic = simulate_spike_dynamics(area_factor=1, stimulus=10e-8)
        self.assertEqual(len(voltage_dynamic), 4001)
        if SHOW_PLOT:
            fig, ax = plt.subplots()
            ax.plot(voltage_dynamic)
            ax.set_xlabel('t(ms)')
            ax.set_ylabel('Voltage(mV)')
            plt.title("Dynamic of one neuron")
            plt.show()

    def test_model_convergence(self):
        """test the convergence of the HH model (based on multiple simulation run)"""
        runs = 10
        counts, sim = HH_model(area_factor=1, stimulus=1e-7, runs=runs, return_seq=True, t_end=2)
        self.assertTrue(all([len(sim_i) > 1 for sim_i in sim]))
        if SHOW_PLOT:
            rows = int(math.sqrt(runs))
            fig, ax = plt.subplots(rows, rows)
            fig.suptitle("Dynamic of all simulations used by the model")
            for i, ax_i in enumerate(ax.flatten()):
                ax_i.plot(sim[i])
            plt.show()

    def test_standardized_model(self):
        """test the convergence of the HH model (based on multiple simulation run)"""
        counts = hh_model_normalised(
            percentage_of_stimulus=1,
            delta_v=0,
            delta_area=0,
            runs=10,
            t_end=2,
            v_ref=1e-7,
            area_ref=0.1
        )
        self.assertTrue(counts > 0)


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.x_data = np.array([10, 20, 30])
        self.y_data = np.array([100, 200, 300])

    def test_loss_computation(self):
        """test that if the model matches the data the error is 0"""
        mse0 = mse_loss(
            parameters=(10, 0),
            model=linear_model,
            x_ref=self.x_data,
            y_ref=self.y_data,
        )
        self.assertEqual(0, mse0)

    def test_optimizer_linear_model(self):
        """test that the optimizer can refine the parameter of the initial guess"""

        def callback_print(x):
            print("stimulus: {:.5f}, area:{:.5f}".format(x[0], x[1]))

        x0 = np.array([1, 1])
        optimum = minimize(
            mse_loss,
            x0,
            args=(linear_model, self.x_data, self.y_data),
            callback=callback_print
        )
        stimulus_opt, area_factor_opt = optimum['x'][0], optimum['x'][1]
        self.assertEqual(int(stimulus_opt), 9)
        self.assertEqual(int(area_factor_opt), 0)

        if SHOW_PLOT:
            y_predicted = np.array([linear_model(x, x0[0], x0[1]) for x in self.x_data])
            y_predicted_opt = np.array([linear_model(x, stimulus_opt, area_factor_opt) for x in self.x_data])
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel('input')
            ax.set_ylabel('output predictions')
            ax.plot(self.x_data, self.y_data, marker='o', markersize=10)
            ax.plot(self.x_data, y_predicted, marker='*')
            ax.plot(self.x_data, y_predicted_opt, marker='*')
            ax.legend(["raw data points", "initial guess", "predictions with optimized parameters"])
            plt.show()


class TestParameterOptimization(unittest.TestCase):

    def test_optimization(self):
        """test that the optimization run without errors.
        reduced runs of simulation and reduced dataset is used"""
        # reference data
        x_data = np.array([0.92, 1, 1.09])
        y_data = np.array([150, 500, 840]) / 1000
        # parameters
        x0 = np.array([0, 0])
        runs = 5
        t_end = 2
        optimum = minimize(
            mse_loss,
            x0,
            args=(hh_model_normalised, x_data, y_data, runs, t_end),
            options={"disp": True},
            bounds=((2, 2), (2, 2)),
        )

    def test_grid_search(self):
        run_grid_search_simulation.main([
            "--runs=1",
            "--voltage-points=10",
            "--voltage-exp-star=-7.5",
            "--voltage-exp-end=-6.5",
            "--area-points=1",
            "--area-exp-start=-1",
            "--area-exp-end=-1",
            "--k-noise-points=5",
            "--k-noise-exp-start=-3",
            "--k-noise-exp-end=-7",
            "--store-results",
            "--store-plots"
        ])


if __name__ == "__main__":
    unittest.main()
