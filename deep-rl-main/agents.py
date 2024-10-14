import random
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class Agent(ABC):
    """A class that defines the basic interface for Deep RL agents (discrete)"""

    @abstractmethod
    def act(self, s: torch.Tensor) -> int:
        """Selects an action for the given state

        :param s: The state to select an action for
        :type s: torch.Tensor
        :raises NotImplementedError: Method must be implemented by concrete agent classes
        :return: An action (discrete)
        :rtype: int
        """
        raise NotImplementedError


class DiscreteActorCriticAgent(Agent):
    """An actor-critic agent that acts on discrete action spaces

    :param num_features: The number of features of the state vector
    :type num_features: int
    :param num_actions: The number of actions available
    :type num_actions: int
    :param device: The device (GPU or CPU) to use
    :type device: torch.device
    """

    def __init__(self, num_features: int, num_actions: int, device: torch.device) -> None:
        # Architecture suggested in the paper "Benchmarking Deep Reinforcement Learning for Continuous Control"
        # https://arxiv.org/pdf/1604.06778.pdf
        self._pi = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(in_features=num_features, out_features=400),
            nn.ReLU(inplace=True),
            # Hidden Layer 2
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU(inplace=True),
            # Output layer
            nn.Linear(in_features=300, out_features=num_actions),
            nn.Softmax(dim=-1),
        ).to(device)

        self._v = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(in_features=num_features, out_features=400),
            nn.ReLU(inplace=True),
            # Hidden Layer 2
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU(inplace=True),
            # Output layer
            nn.Linear(in_features=300, out_features=1),
        ).to(device)

        self._device = device

    @property
    def pi(self) -> nn.Module:
        """The policy function approximator

        :return: The policy approximator as a PyTorch module
        :rtype: nn.Module
        """
        return self._pi

    @property
    def v(self) -> nn.Module:
        """The value function approximator

        :return: The value approximator as a PyTorch module
        :rtype: nn.Module
        """
        return self._v

    def act(self, s: torch.Tensor) -> int:
        return torch.distributions.Categorical(self._pi(s.to(self._device))).sample().item()

    def step(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs one single step

        Computes the value of the given states and the log probabilities, pi(a|s), of the given actions

        :param s: The states to compute values for
        :type s: torch.Tensor
        :param a: The actions to compute log probabilities for
        :type a: torch.Tensor
        :return: A tuple of V(s) (batch_size,) and pi(a|s) (batch_size,)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        s = s.to(self._device)
        probs = self._pi(s)
        distribution = torch.distributions.Categorical(probs)
        values = self._v(s)

        # pylint: disable=not-callable
        return values.squeeze(), distribution.log_prob(a.to(self._device))


class ContinuousActorCriticAgent(Agent):
    """An actor-critic agent that acts on continuous action spaces

    :param num_features: The number of features of the state vector
    :type num_features: int
    :param action_dim: The dimension of the action space (i.e. 1-D, 2-D, etc.)
    :type action_dim: int
    :param device: The device (GPU or CPU) to use
    :type device: torch.device
    """

    def __init__(self, num_features: int, action_dim: int, device: torch.device) -> None:
        # Architecture suggested in the paper "Continuous Control with Deep Reinforcement Learning"
        # https://arxiv.org/pdf/1509.02971.pdf
        self._pi = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(in_features=num_features, out_features=400),
            nn.ReLU(inplace=True),
            # Hidden Layer 2
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU(inplace=True),
            # Output layer
            nn.Linear(in_features=300, out_features=action_dim),
            nn.Tanh(),
        ).to(device)

        self._q = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(in_features=num_features + action_dim, out_features=400),
            nn.ReLU(inplace=True),
            # Hidden Layer 2
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU(inplace=True),
            # Output layer
            nn.Linear(in_features=300, out_features=1),
        ).to(device)

        self._device = device

    @property
    def pi(self) -> nn.Module:
        """The policy function approximator

        :return: The policy approximator as a PyTorch module
        :rtype: nn.Module
        """
        return self._pi

    @property
    def q(self) -> nn.Module:
        """The Q function approximator

        :return: The Q approximator as a PyTorch module
        :rtype: nn.Module
        """
        return self._q

    def act(self, s: torch.Tensor) -> torch.Tensor:
        # We need `torch.no_grad()` because we are going to be returning a tensor
        # as opposed to integers (like in the discrete agent) which means we need
        # to make sure the tensor is not added to the computational graph
        # This is not needed in the discrete case because we call `item()` which
        # automatically returns an integer which isn't part of the graph
        # We also always move the tensor to the cpu because acting in the Gym
        # environments can't be done in the GPU
        # Once again this is not needed when using `item()` (discrete) because
        # the integer is returned already in the CPU
        with torch.no_grad():
            return self._pi(s.to(self._device)).cpu()


class QAgent(Agent):
    """A Q-agent that acts on discrete action spaces

    :param num_features: The number of features of the state vector
    :type num_features: int
    :param num_actions: The number of actions available
    :type num_actions: int
    :param initial_epsilon: The initial value for epsilon
    :type initial_epsilon: float
    :param final_epsilon: The final value for epsilon
    :type final_epsilon: float
    :param epsilon_decay: The decay factor for annealing epsilon linearly
    :type epsilon_decay: float
    :param device: The device (GPU or CPU) to use
    :type device: torch.device
    """

    def __init__(
        self,
        num_features: int,
        num_actions: int,
        initial_epsilon: float,
        final_epsilon: float,
        epsilon_decay: float,
        device: torch.device,
    ) -> None:
        # Following a similar architecture as the actor-critic agent
        self._q = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(in_features=num_features, out_features=400),
            nn.ReLU(inplace=True),
            # Hidden Layer 2
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU(inplace=True),
            # Output layer
            nn.Linear(in_features=300, out_features=num_actions),
            # No softmax since we're outputting action values not probabilities
        ).to(device)

        self._num_actions = num_actions
        self._initial_epsilon = initial_epsilon
        self._final_epsilon = final_epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon = initial_epsilon
        self._device = device
        self._explore = True

    @property
    def q(self) -> nn.Module:
        """The Q-function approximator

        :return: The Q-function approximator as a PyTorch module
        :rtype: nn.Module
        """
        return self._q

    def act(self, s: torch.Tensor) -> int:
        if random.random() < self._epsilon:
            action = random.choice(list(range(self._num_actions)))
        else:
            action = torch.argmax(self._q(s.to(self._device))).item()

        # Linearly anneal epsilon like in the original DQN paper
        # https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        self._epsilon -= (self._initial_epsilon - self._final_epsilon) / float(self._epsilon_decay)

        return action

    def step(self, s: torch.Tensor) -> torch.Tensor:
        """Performs one single step

        Computes the action values of the given state(s), for all available actions

        :param s: The state(s) to compute action values for
        :type s: torch.Tensor
        :return: The actions values of shape (n_states, n_actions)
        :rtype: torch.Tensor
        """
        return self._q(s.to(self._device))

    def explore(self, do_explore: bool) -> None:
        """Sets whether this agent should explore or not (i.e. purely greedy)

        :param do_explore: Whether to explore or not
        :type do_explore: bool
        """
        self._explore = do_explore
