import torch
import torch.nn as nn


class Neural_Net(nn.Module):
    def __init__(
        self,
        units,
        activation,
        input=1,
        output=1,
        create_parameter=False,
        n_paramater=1,
        dropout_prob=0.0,
    ):
        super().__init__()
        self.units = units
        self.output = output
        self.create_parameter = create_parameter
        self.n_paramater = n_paramater
        self.dropout_prob = dropout_prob

        self.hidden_layers = nn.ModuleList([nn.Linear(input, units[0])])

        self.hidden_layers.extend(
            [nn.Linear(units[_], units[_ + 1]) for _ in range(len(self.units) - 1)]
        )
        # Última camada linear
        self.output_layer = nn.Linear(units[-1], self.output)

        if not isinstance(activation, list):
            self.activation = [activation] * len(self.units)

        else:
            self.activation = activation

        # Camadas de Dropout
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(len(self.units))]
        )

        if self.create_parameter:
            self.param = nn.Parameter(
                torch.rand(self.n_paramater), requires_grad=True
            )  # * 2 * np.pi)

    def forward(self, x):
        for layer, activation, dropout in zip(
            self.hidden_layers, self.activation, self.dropouts
        ):
            x = activation(layer(x))
            x = dropout(x)
        x = self.output_layer(x)
        return x
