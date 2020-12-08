# neuron_spike_simulation
Simulation and parameter optimization of HH model 
for spike counting of a neuron excited by a voltage input

Note: the code of the hh model has been imported from:   
https://stackoverflow.com/questions/65159861/grid-search-over-function/  
it has then been augmented to answer the question.

## Design
The framework contains:
- an implementation of the HH model     
- a script to compute and plot the predictions of 
how many spikes a neuron does, as function of the input stimulus and the surface area of an electrode
- a script to search the values of the input stimulus and the surfaces area that 
makes the model to fit the experimental data

## Installation
Before to start:  
Add the current project folder path to PYTHONPATH.  
In ~/.bashrc, append:
```
PYTHONPATH=your/path/to/repo:$PYTHONPATH 
export PYTHONPATH
```
e.g.
```
PYTHONPATH=~/PycharmProjects/stack-overflow-contrib:$PYTHONPATH 
export PYTHONPATH
```

To install and activate the environment:
```
conda env create -f neuron_spike_simulation/environment.yml
conda activate neuron_spike_simulation
```


## Usage
to search the values that generate a model that fits the experimental data, run:
```
python neuron_spike_simulation/script/optimise_parameters.py
```
to predict the response of a neuron given a percentage variation
of the parameters (voltage and area), run:
```
python neuron_spike_simulation/script/predict_and_plot.py --delta-v=0 --delta-area=0 --runs=10
```

## Tests
To run the tests:
```
python -m unittest discover neuron_spike_simulation/src
```

## Contribution:
In order to make the optimization to converge,
few options have been investigated, such as:
- normalising the input and the output of the model
- optimize the variations of the parameters respect to a fixed values
However, the optimisation still cannot make the simulated model
to converge to the experimental data.

Possible issues:
- Optimizer settings
- Sensitivity of the loss respect to the variation of the parameters
- The simulation has been treated as a black box, it might be incorrect 