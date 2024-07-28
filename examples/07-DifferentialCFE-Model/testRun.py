import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run
import os
print("Current working directory:", os.getcwd())

# by default we assume that you have at least one CUDA-capable NVIDIA GPU
if torch.cuda.is_available():
    print("GPU")
    start_run(config_file=Path("basin_cudalstmHourly.yml"))

# fall back to CPU-only mode
else:
    print("CPU")
    start_run(config_file=Path("basin_cudalstmHourly.yml"), gpu=-1)