import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple Multi-layer perceptron model.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Multi-layer perceptron initialization.
        :param input_size: number of input features
        :param output_size: number of output features
        """
        super(MLP, self).__init__()
        self._layers = nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size)
        )

        self.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-layer perceptron.
        :param x: input tensor of shape (batch_size, input_size)
        :return: output tensor of shape (batch_size, output_size)
        """
        return self._layers(x)