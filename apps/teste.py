import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.train_and_eval import train, train_with_parameter
from src.visualization.plot_functions import (
    plot_expected_values,
    plot_states,
    plot_loss_functions,
    plot_learned_param,
)

from src.data_simulation.jaynes_cummings_data import data_jc
import numpy as np
import random

params={"wc": 1., "wa": 1.0, "g": 0.1}#random.uniform(0.1, 1.0)
tfinal = 15
n_time_steps = 100
init_state = "coherent"
dims = {"atom": 2, "field": 2}
picture = "full"
epochs = 100

# PARA APRENDER O ESTADO E O PARÂMETRO

models_dict, loss_dict = train_with_parameter(
    epochs      = epochs,
    params      = params,
    tfinal      = tfinal,
    n_time_steps= n_time_steps,
    init_state  = init_state,
    picture     = picture,
    dims        = dims,
    n_points_loss=100,
)


import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

epochs = len(loss_dict["total_loss"])
skip_epochs = int(epochs // 1)

plt.style.use('science')
plt.figure(figsize=(10, 6))
plt.plot(loss_dict["learned_param"][0:-1:skip_epochs], label="Learned Parameter")
plt.xlabel("Epochs")
plt.ylabel("Parameter Value")
plt.title("Learned Parameter During Training")
plt.legend()
plt.grid()
plt.show()