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
            # nn.Sequential(OrthoLinear(32, 16), nn.ReLU())
        ])
        # self.output: nn.Linear = nn.Linear(16, config["output_size"])

        task_input_size = 32
        task_output_size = 5

        # task specific layers
        self.task0 = OrthoLinear(task_input_size, task_output_size)
        self.task1 = OrthoLinear(task_input_size, task_output_size)
        self.task2 = OrthoLinear(task_input_size, task_output_size)
        self.task3 = OrthoLinear(task_input_size, task_output_size)
        self.task4 = OrthoLinear(task_input_size, task_output_size)
        self.task5 = OrthoLinear(task_input_size, task_output_size)
        self.task6 = OrthoLinear(task_input_size, task_output_size)
        self.task7 = OrthoLinear(task_input_size, task_output_size)
        self.task8 = OrthoLinear(task_input_size, task_output_size)
        self.task9 = OrthoLinear(task_input_size, task_output_size)
        self.task10 = OrthoLinear(task_input_size, task_output_size)
        self.task11 = OrthoLinear(task_input_size, task_output_size)
        self.task12 = OrthoLinear(task_input_size, task_output_size)
        self.task13 = OrthoLinear(task_input_size, task_output_size)

    def forward(self, x: torch.tensor, task_id : int):
        """
        Args:
            x (torch.tensor): Shape[batch_size, input_size]

        Returns:
            _type_: _description_
        """
        embeds: torch.tensor = self.embeds(x)
        for i, layer in enumerate(self.linearlayers):
            embeds: torch.tensor = layer(embeds)
            
        # output: torch.tensor = self.output(embeds)
        if task_id == 0:
            output: torch.tensor = self.task0(embeds)
        elif task_id == 1:
            output: torch.tensor = self.task1(embeds)
        elif task_id == 2:
            output: torch.tensor = self.task2(embeds)
        elif task_id == 3:
            output: torch.tensor = self.task3(embeds)
        elif task_id == 4:
            output: torch.tensor = self.task4(embeds)
        elif task_id == 5:
            output: torch.tensor = self.task5(embeds)
        elif task_id == 6:
            output: torch.tensor = self.task6(embeds)
        elif task_id == 7:
            output: torch.tensor = self.task7(embeds)
        elif task_id == 8:
            output: torch.tensor = self.task8(embeds)
        elif task_id == 9:
            output: torch.tensor = self.task9(embeds)
        elif task_id == 10:
            output: torch.tensor = self.task10(embeds)
        elif task_id == 11:
            output: torch.tensor = self.task11(embeds)
        elif task_id == 12:
            output: torch.tensor = self.task12(embeds)
        elif task_id == 13:
            output: torch.tensor = self.task13(embeds)
        else:
            assert False, 'Bad Task ID passed'
        
        return torch.softmax(output, dim=1)  # Apply softmax activation

