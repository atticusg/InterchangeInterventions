import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from layered_intervenable_model import LayeredIntervenableModel
import utils

__author__ = "Atticus Geiger"
__version__ = "CS224u, Stanford, Spring 2022"


class ActivationLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_activation):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, device=device)
        self.activation = hidden_activation

    def forward(self, x):
        return self.activation(self.linear(x))


class LIMDeepNeuralClassifier(LayeredIntervenableModel):
    def __init__(self,
            hidden_dim=50,
            hidden_activation=nn.Tanh(),
            num_layers=1,
            input_dim=None,
            n_classes=None,
            device=None):
        """
        A layered interventable model

        h_1 = f(xW_1 + b_1)
        ...
        h_k = f(xW_k + b_k)
        ...
        y = softmax(h_nW_y + b_y)

        with a cross-entropy loss and f determined by `hidden_activation`.

        Parameters
        ----------
        hidden_dim : int
            Dimensionality of the hidden layer.

        hidden_activation : nn.Module
            The non-activation function used by the network for the
            hidden layer.

        input_dim : int
            Dimensionality of the input layer.

        n_classes : int
            Dimensionality of the output.

        **base_kwargs
            For details, see `torch_model_base.py`.

        Attributes
        ----------
        loss: nn.CrossEntropyLoss(reduction="mean")

        self.params: list
            Extends TorchModelBase.params with names for all of the
            arguments for this class to support tuning of these values
            using `sklearn.model_selection` tools.

        """
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_classes_ = n_classes
        self.hidden_activation = hidden_activation
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.model_layers = torch.nn.ModuleList()
        self.model_layers.append(
            ActivationLayer(
                self.input_dim,
                self.hidden_dim,
                self.device,
                self.hidden_activation))
        self.dims = [self.input_dim,self.hidden_dim]
        # Hidden to hidden:
        for i in range(self.num_layers-1):
            self.dims.append(self.hidden_dim)
            self.model_layers.append(
                ActivationLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.device,
                    self.hidden_activation))
        # Hidden to output:
        self.model_layers.append(
            nn.Linear(self.hidden_dim, self.n_classes_, device=self.device))
        self.dims.append(self.n_classes_)
        self.build_graph(self.model_layers, self.dims)
