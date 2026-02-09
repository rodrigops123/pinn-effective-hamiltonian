# Introduction
This is a code base used to train a neural network based on the Physics Informed Neural Networks (PINN) framework to:
- learn the vector state associated with a Jaynes-Cummings model
- learn the effective Hamiltonian of a Jaynes-Cummings model

# Code structure
Below there is a schematic view of the code structure of this project:

```
|___apps/
|   |___main.py
|___config/
|   |___global_variables.py
|___notebooks/
|___src/
|   |___data_simulation/
|   |   |___jaynes_cummings_data.py
|   |___model/
|   |   |___loss_functions.py
|   |   |___neural_network.py
|   |   |___train_and_eval.py
|   |___visualization/
|       |___plot_functions.py
|___requirements.txt
|___utils.py
```
