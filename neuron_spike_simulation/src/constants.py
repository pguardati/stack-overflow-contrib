import os

PROJECT_NAME = "neuron_spike_simulation"
REPOSITORY_PATH = os.path.realpath(__file__)[:os.path.realpath(__file__).find(PROJECT_NAME)]
PROJECT_PATH = os.path.join(REPOSITORY_PATH, PROJECT_NAME)
LOG_DIR = os.path.join(PROJECT_PATH, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
print("Project Path:", PROJECT_PATH)
