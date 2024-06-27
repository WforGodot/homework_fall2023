import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        obs = ptu.from_numpy(obs)
        action = self.forward(obs).sample().cpu().numpy()

        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            logits = self.logits_net(obs)
            return distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(obs)
            return distributions.Normal(mean, torch.exp(self.logstd))
        

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        self.optimizer.zero_grad()
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        dist = self.forward(obs)

        # Compute log probabilities and handle continuous action spaces
        log_prob = dist.log_prob(actions)
        if not self.discrete:
            log_prob = log_prob.sum(dim=-1)

        loss = -(log_prob).mean()
        loss.backward()
        self.optimizer.step()

        return {
            "Training Loss": ptu.to_numpy(loss),
        }

class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        self.optimizer.zero_grad()
        dist = self.forward(obs)
        
        # Compute log probabilities and ensure the shape matches advantages
        log_prob = dist.log_prob(actions)
        
        # For continuous action spaces, sum the log probabilities along the action dimension
        if not self.discrete:
            log_prob = log_prob.sum(dim=-1)
        
        # Calculate the loss
        loss = -(log_prob * advantages).mean()
        
        # Perform backpropagation and optimization step
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }