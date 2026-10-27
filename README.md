# Uso do MixFunn
- A implementação do MixFunn está nos caminhos `src/model/mixfunn.py` e `src/model/models.py`.
- O script `src/model/mixfunn.py` implementa o mixfunn de fato, com o neurônio de segunda ordem, com as combinações lineares de função de ativação etc.
- O script `src/model/models.py` instancia o mixfunn numa rede. Ele usa os blocos que foram definidos na `mixfunn.py` para gerar uma rede. Essa rede que é instanciada na hora do treino. Foi nesse script, na classe MixFunn onde eu inseri os parâmetros do sistema.

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
|___scripts/
|   |___migrate_pickles.py
|___tests/
|___pyproject.toml
|___utils.py
```

# Setup
```
pip install ".[dev]"
pytest
```
`tests/test_golden.py` pins the current numerical behaviour; regenerate it with
`UPDATE_GOLDEN=1 pytest tests/test_golden.py` only after an intentional change.
Known bugs are documented as strict `xfail` tests.

Training results are not tracked in git. Legacy `.pkl` results can be converted to a
format that survives refactoring with `python scripts/migrate_pickles.py`.
