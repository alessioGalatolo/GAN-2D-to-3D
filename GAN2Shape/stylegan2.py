# FIXME: This is a placeholder for the stylegan implementation

import torch
import torch.nn as nn


class StyleGAN2(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.single_layer = nn.Linear(input_size, output_size)
        self.single_activation = nn.ReLU()

    def forward(self, input):
        output = self.single_layer(input)
        return self.single_activation(output)
