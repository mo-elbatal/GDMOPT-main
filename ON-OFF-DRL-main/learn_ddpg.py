import argparse
import random
from collections import namedtuple
from copy import deepcopy
from typing import List, Tuple

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# For reproducibility
torch.manual_seed(24)
random.seed(24)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

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


def evaluate(env: gym.Env, agent: Agent, episodes: int, verbose: bool) -> None:
    """Evaluates the agent by interacting with the environment and produces a plot of the rewards

    :param env: The environment to interact with
    :type env: gym.Env
    :param agent: The agent to evaluate
    :type agent: Agent
    :param episodes: The episodes to interact
    :type episodes: int
    :param verbose: Whether to run in verbose mode or not
    :type verbose: bool
    """
    rewards = []

    for _ in tqdm(range(episodes), disable=not verbose):
        s = env.reset()
        done = False
        reward = 0.0

        while not done:
            s = torch.from_numpy(s).float()
            a = agent.act(s)
            s_prime, r, done, _ = env.step(a)
            reward += r
            s = s_prime

        rewards.append(reward)

    print(f"Mean reward over {episodes} episodes: {np.mean(rewards)}")


def plot_rewards(rewards: List[float], title: str, output_dir: str, filename: str) -> None:
    """Plots the given rewards per episode

    :param rewards: The rewards to plots, assumed to be one per _episode_
    :type rewards: List[float]
    :param title: The title for the plot
    :type title: str
    :param output_dir: str
    :type output_dir: The directry where the plot will be saved to (will be created if it doesn't exist)
    :param filename: The filename for the plot without `.png`
    :type filename: str
    """
    Path(f"./output/{output_dir}").mkdir(exist_ok=True)
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(f"./output/{output_dir}/{filename}.png")


def render_interaction(env: gym.Env, agent: Agent, output_dir: str, filename: str) -> None:
    """Renders an interaction producing a GIF file

    Assumes `ffmpeg` has been installed in the system

    :param env: The environment to interact with
    :type env: gym.Env
    :param agent: The agent that interacts with the environment
    :type agent: Agent
    :param output_dir: str
    :type output_dir: The directry where the plot will be saved to (will be created if it doesn't exist)
    :param filename: The name of the output file without `.gif`
    :type filename: str
    """
    Path(f"./output/{output_dir}").mkdir(exist_ok=True)

    frames = []
    s = env.reset()
    done = False
    reward = 0.0

    while not done:
        frames.append(env.render(mode="rgb_array"))

        s = torch.from_numpy(s).float()
        a = agent.act(s)
        s_prime, r, done, _ = env.step(a)
        reward += r
        s = s_prime

    env.close()
    print(f"Total reward from interaction: {reward}")
    _to_gif(frames, f"{output_dir}/{filename}")


def _to_gif(frames: List[np.ndarray], filename: str, size: Tuple[int, int] = (72, 72), dpi: int = 72) -> None:
    print(f"Generating GIF: {filename}.gif")
    plt.figure(figsize=(frames[0].shape[1] / size[0], frames[0].shape[0] / size[1]), dpi=dpi)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(f"./output/{filename}.gif", writer="ffmpeg", fps=60)


class Buffer:
    """Experience replay buffer

    :param capacity: The max capacity of the buffer
    :type capacity: int
    """

    def __init__(self, capacity: int) -> None:
        self._max_capacity = capacity
        self._buf = []
        self._capacity = 0

    def save(self, experience: Experience) -> None:
        """Saves the given experience to the buffer

        When the max capacity is reached, an old experience is removed in a FIFO way.

        :param experience: The experience to save
        :type experience: Experience
        """
        if self._capacity == self._max_capacity:
            self._buf.pop(0)
            self._buf.append(experience)
        else:
            self._buf.append(experience)
            self._capacity += 1

    def get(self, batch_size: int) -> List[Experience]:
        """Gets a random batch of experiences from the buffer

        :param batch_size: The size of the batch to get
        :type batch_size: int
        :return: A list of experiences
        :rtype: List[Experience]
        """
        return random.choices(self._buf, k=batch_size)


# pylint: disable=too-many-locals
def ddpg(
    env: gym.Env,
    agent: ContinuousActorCriticAgent,
    epochs: int,
    max_steps: int,
    buffer_capacity: int,
    batch_size: int,
    alpha: float,
    gamma: float,
    polyak: float,
    act_noise: float,
    verbose: bool,
) -> List[float]:
    """Trains an agent using Deep Deterministic Policy Gradients algorithm

    :param env: The environment to train the agent in
    :type env: gym.Env
    :param agent: The agent to train
    :type agent: ContinuousActorCriticAgent
    :param epochs: The number of epochs to train the agent for
    :type epochs: int
    :param max_steps: The max number of steps per episode
    :type max_steps: int
    :param buffer_capacity: Max capacity of the experience replay buffer
    :type buffer_capacity: int
    :param batch_size: Batch size to use of experiences from the buffer
    :type batch_size: int
    :param gamma: The discount factor
    :type gamma: float
    :param alpha: The learning rate
    :type alpha: float
    :param polyak: Interpolation factor in polyak averaging for target networks
    :type polyak: float
    :param act_noise: Standard deviation for Gaussian exploration noise added to policy at training time
    :type act_noise: float
    :param verbose: Whether to run in verbose mode or not
    :type verbose: bool
    :return: The total reward per episode
    :rtype: List[float]
    """
    pi_optimizer = optim.Adam(agent.pi.parameters(), lr=alpha)
    q_optimizer = optim.Adam(agent.q.parameters(), lr=alpha)
    target_pi = deepcopy(agent.pi).to(device)
    target_q = deepcopy(agent.q).to(device)
    experience_buf = Buffer(buffer_capacity)
    total_rewards = []

    for _ in tqdm(range(epochs), disable=not verbose):
        observation, _ = env.reset()  # Only take the observation part
        s = torch.from_numpy(np.array(observation)).float()
        done = False
        reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            # Collect and save experience from the environment
            # Add Gaussian noise to the action for exploration
            a = agent.act(s) + torch.normal(mean=0.0, std=act_noise, size=(1,))
            a = a.item()
            s_prime, r, done, _, _ = env.step(a)
            s_prime = torch.from_numpy(s_prime).float()

            reward += r
            experience_buf.save(Experience(s, a, r, s_prime, done))

            # Learn from previous experiences
            experiences = experience_buf.get(batch_size)
            loss = 0.0

            states = torch.stack([e.state for e in experiences]).to(device)
            actions = torch.stack([e.action for e in experiences]).to(device)
            rewards = [e.reward for e in experiences]
            next_states = torch.stack([e.next_state for e in experiences]).to(device)
            dones = [e.done for e in experiences]

            q_values = agent.q(torch.cat([states, actions], dim=-1))
            next_qvalues = target_q(torch.cat([next_states, target_pi(next_states)], dim=-1))
            # Keep a copy of the current Q-values to be used for the TD targets
            td_targets = q_values.clone()

            # Compute TD targets
            for index in range(batch_size):
                # Terminal states do not have a future value
                if dones[index]:
                    next_qvalues[index] = 0.0

                td_targets[index] = rewards[index] + gamma * next_qvalues[index]

            # Compute TD error and loss (MSE)
            loss = (td_targets - q_values) ** 2
            loss = loss.mean()
            # Update the value function
            q_optimizer.zero_grad()
            loss.sum().backward()
            q_optimizer.step()

            # Update the policy
            # We use the negative loss because policy optimization is done using gradient _ascent_
            # This is because in policy gradient methods, the "loss" is a performance measure that is _maximized_
            loss = -agent.q(torch.cat([states, agent.pi(states)], dim=-1))
            loss = loss.mean()
            pi_optimizer.zero_grad()
            loss.backward()
            pi_optimizer.step()

            # Update target networks with polyak averaging
            with torch.no_grad():
                for target_p, p in zip(target_pi.parameters(), agent.pi.parameters()):
                    target_p.copy_(polyak * target_p + (1.0 - polyak) * p)

            with torch.no_grad():
                for target_p, p in zip(target_q.parameters(), agent.q.parameters()):
                    target_p.copy_(polyak * target_p + (1.0 - polyak) * p)

            s = s_prime
            steps += 1

        total_rewards.append(reward)

    return total_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute Deep Deterministic Policy Gradients against Pendulum-v1 environment"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to train")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--buf-capacity", type=int, default=50000, help="Max capacity of the experience replay buffer")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size to use of experiences from the buffer")
    parser.add_argument("--alpha", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument(
        "--polyak", type=float, default=0.9, help="Interpolation factor in polyak averaging for target networks"
    )
    parser.add_argument(
        "--act-noise", type=float, default=0.2, help="Standard deviation for Gaussian exploration noise"
    )
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes to use for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode")
    parser.add_argument("--save-gif", action="store_true", help="Save a GIF of an interaction after training")
    args = parser.parse_args()

    agent = ContinuousActorCriticAgent(
        num_features=3,
        action_dim=1,
        device=device,
    )
    env = gym.make("Pendulum-v1")
    # For reproducibility
    obs, info = env.reset(seed=24)

    print(f"Training agent with the following args\n{args}")
    rewards = ddpg(
        env,
        agent,
        epochs=args.epochs,
        max_steps=args.max_steps,
        buffer_capacity=args.buf_capacity,
        batch_size=args.batch_size,
        alpha=args.alpha,
        gamma=args.gamma,
        polyak=args.polyak,
        act_noise=args.act_noise,
        verbose=args.verbose,
    )

    plot_rewards(rewards, title="DDPG on Pendulum-v1", output_dir="ddpg", filename="Pendulum-v1")

    print("Evaluating agent")
    evaluate(env, agent, args.eval_episodes, args.verbose)

    if args.save_gif:
        print("Rendering interaction")
        render_interaction(env, agent, output_dir="ddpg", filename="Pendulum-v1")