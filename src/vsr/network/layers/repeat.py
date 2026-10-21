# Adapted from auto_avsr/ESPnet under Apache 2.0.
# Modified to remove stochastic layer dropping used during training.
"""Sequential layers with multiple inputs and outputs."""

import torch


class MultiSequential(torch.nn.Sequential):
    def forward(self, *args):
        for layer in self:
            args = layer(*args)
        return args


def repeat(count, factory):
    return MultiSequential(*[factory(index) for index in range(count)])
