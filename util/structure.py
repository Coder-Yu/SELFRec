import numpy as np
import torch

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate,activation='relu'):
        super(PointWiseFeedForward, self).__init__()
        act = None
        if activation == 'relu':
            act = torch.nn.ReLU()
        elif activation == 'gelu':
            act = torch.nn.GELU()
        self.pwff = torch.nn.Sequential(
            torch.nn.Linear(hidden_units, hidden_units),
            #torch.nn.Dropout(p=dropout_rate),
            act,
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs):
        outputs = self.pwff(inputs)
        outputs += inputs
        return outputs
