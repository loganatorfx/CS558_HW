from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: nn.Module = nn.Tanh(),
        output_activation: nn.Module = nn.Identity(),
) -> nn.Module:
    """
    Builds a feedforward neural network (MLP)

    :param input_size: Number of input features
    :param output_size: Number of output features
    :param n_layers: Number of hidden layers
    :param size: Number of neurons in each hidden layer
    :param activation: Activation function for hidden layers
    :param output_activation: Activation function for output layer

    :return: A PyTorch Sequential model representing the MLP.
    """
    layers = []
    in_size = input_size

    # Create hidden layers
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))  # Fully connected layer
        layers.append(activation)  # Activation function
        in_size = size  # Next layer input size

    # Output layer
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)  # Apply final activation function

    return nn.Sequential(*layers)  # Return as a PyTorch Sequential model


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
