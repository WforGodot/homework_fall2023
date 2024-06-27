from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # flatten the lists of arrays into single arrays
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            critic_info = {}
            for _ in range(self.baseline_gradient_steps):
                # Update the critic network
                critic_info = self.critic.update(obs, q_values)

            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: trajectory-based PG
            q_values = [self._discounted_return(r) for r in rewards]
        else:
            # Case 2: reward-to-go PG
            q_values = [self._discounted_reward_to_go(r) for r in rewards]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            advantages = q_values
        else:
            # run the critic and use it as a baseline
            values = self.critic.forward(ptu.from_numpy(obs)).cpu().detach().numpy().flatten()
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # advantages without GAE
                advantages = q_values - values
            else:
                # Implement GAE
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size)
                last_adv = 0
                for t in reversed(range(batch_size)):
                    if terminals[t] == 1:
                        delta = rewards[t] - values[t]
                        last_adv = delta  # Reset last advantage if episode ends
                    else:
                        delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                        last_adv = delta + self.gamma * self.gae_lambda * last_adv
                    advantages[t] = last_adv

        # normalize the advantages
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """Helper function to calculate the discounted return."""
        cumulative_sum = 0
        for reward in rewards[::-1]:
            cumulative_sum = reward + self.gamma * cumulative_sum

        return [cumulative_sum] * len(rewards)

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """Helper function to calculate the discounted reward-to-go."""
        cumulative_sum = 0
        discounted_sums = []
        for reward in rewards[::-1]:
            cumulative_sum = reward + self.gamma * cumulative_sum
            discounted_sums.append(cumulative_sum)
        return discounted_sums[::-1]
