import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs558.infrastructure import pytorch_util as ptu
from cs558.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # Ensure obs is a batch (even if it's a single sample)
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]  # Add batch dimension
        
        # Convert to PyTorch tensor and move to GPU (if available)
        observation = ptu.from_numpy(observation)

        # Compute mean action from the network
        mean_action = self.mean_net(observation)

        # Sample from a Gaussian distribution if policy is stochastic
        if not self.discrete:
            std = torch.exp(self.logstd)  # Convert log std to std
            action_distribution = torch.distributions.Normal(mean_action, std)
            action = action_distribution.sample()  # Sample an action
        else:
            # Discrete case: use softmax to sample an action
            action_distribution = torch.distributions.Categorical(logits=self.logits_na(observation))
            action = action_distribution.sample()

        # Convert action to numpy and return
        return ptu.to_numpy(action)


    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        # Convert NumPy array to PyTorch tensor if necessary
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)  # Convert using helper function

        mean_action = self.mean_net(observation)  # Now it's guaranteed to be a tensor

        if self.discrete:
            action_distribution = torch.distributions.Categorical(logits=self.logits_na(observation))
        else:
            std = torch.exp(self.logstd)
            action_distribution = torch.distributions.Normal(mean_action, std)

        return action_distribution


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(self, observations, actions, adv_n=None, acs_labels_na=None, qvals=None):
        """
        Updates the policy by computing loss and performing backpropagation.
        """
        # Forward pass – Get predicted action distribution
        action_distribution = self.forward(observations)

        # Extract the mean action
        if isinstance(action_distribution, torch.distributions.Normal):
            predicted_actions = action_distribution.mean
        else:
            predicted_actions = action_distribution

        # # Debugging prints
        # print(f"predicted_actions type: {type(predicted_actions)}, shape: {predicted_actions.shape}")
        # print(f"actions type: {type(actions)}, shape: {actions.shape}")

        # Ensure actions is a torch tensor
        if isinstance(actions, np.ndarray):
            actions = ptu.from_numpy(actions)
        
        # Compute loss
        loss = self.loss(predicted_actions, actions)
        

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

