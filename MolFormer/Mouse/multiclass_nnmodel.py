from typing import OrderedDict
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class OrthoLinear(torch.nn.Linear):
    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class XavierLinear(torch.nn.Linear):
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class NNModel(nn.Module):
    def __init__(self, config):
        """Instantiates NN linear model with arguments from

        Args:
            config (args): Model Configuration parameters.
        """
        self.loss_fxn = config['loss_fxn']
        super(NNModel, self).__init__()
        self.embeds: nn.Sequential = nn.Sequential(
            nn.Linear(config["input_size"], config["emb_size"]),
            nn.ReLU(),
            OrthoLinear(config["emb_size"], config["hidden_size"]),
            nn.ReLU(),
        )

        self.linearlayers: nn.ModuleList = nn.ModuleList([
            nn.Sequential(OrthoLinear(config["hidden_size"], 256), nn.ReLU()),
            nn.Sequential(OrthoLinear(256, 128), nn.ReLU()),
            nn.Sequential(OrthoLinear(128,64), nn.ReLU()),
            nn.Sequential(OrthoLinear(64, 32), nn.ReLU()),
            nn.Sequential(OrthoLinear(32, 32), nn.ReLU()),
        ])
        
        self.output: nn.Linear = nn.Linear(32, config["output_size"])

    
    def forward(self, x: torch.tensor):
        """
        Args:
            x (torch.tensor): Shape[batch_size, input_size]

        Returns:
            _type_: _description_
        """
        embeds: torch.tensor = self.embeds(x)
        for i, layer in enumerate(self.linearlayers):
            embeds: torch.tensor = layer(embeds)
        output: torch.tensor = self.output(embeds)
        
        return output

