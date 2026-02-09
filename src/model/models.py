import torch
import torch.nn as nn
from .mixfunn import Mixfun, Quad


class MixFunn(nn.Module):

    def __init__(
        self,
        input_=1,
        output_=1,
        create_parameter=False,
        n_paramater=1,
        p_drop=False,
    ):
        super(MixFunn, self).__init__()
        n = 32
        self.layer = nn.Sequential(
            Mixfun(
                input_,
                # n,
                output_,
                second_order_input=True,
                second_order_function=True,
                p_drop=p_drop,
            ),
            # mf.Quad(n,n, second_order = False),
            # Quad(n, output_, second_order=True),
        )
        
        if create_parameter:
            self.param = nn.Parameter(
                torch.rand(n_paramater), requires_grad=True
            )

        total_params = sum(p.numel() for p in self.parameters())

        print(f"Parameter total: {total_params}")

    def forward(self, x):
        x = self.layer(x)
        return x
