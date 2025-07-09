import torch.nn as nn
import torch.nn.functional as F
import math
import mup
from mup import MuReadout, set_base_shapes

class MLP_mup(nn.Module):
    def __init__(self, in_dim, out_dim, width, n_hidden, activation=F.relu):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, width)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(n_hidden)])
        self.out_layer = MuReadout(width, out_dim)
        self.activation = activation


    def forward(self, x):
        x = self.in_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.out_layer(x)
        return x

    def zero_output(self):
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.in_layer.bias)
        # self.out_layer.bias.zero_()

def get_MLP_mup(in_dim, out_dim, width, n_hidden, activation=F.relu, zero_out=False) -> MLP_mup:
    base = MLP_mup(in_dim, out_dim, 1, n_hidden, activation=activation)
    delta = MLP_mup(in_dim, out_dim, 2, n_hidden, activation=activation)
    model = MLP_mup(in_dim, out_dim, width, n_hidden, activation=activation)
    set_base_shapes(model, base, delta=delta)

    for p in model.parameters():
        if p.dim() > 1: # Don't scale bias layers
            mup.init.kaiming_uniform_(p, a=math.sqrt(5))

        # mup.init.uniform_(p, -0.1, 0.1)

    if zero_out:
        model.zero_output()
    return model


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, n_hidden, activation=F.relu):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, width)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(n_hidden)])
        self.out_layer = nn.Linear(width, out_dim)
        self.activation = activation

        self.zero_output()

    def forward(self, x):
        x = self.in_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.out_layer(x)
        return x

    def zero_output(self):
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.in_layer.bias)
        # self.out_layer.bias.zero_()
