import argparse
import random
from collections import namedtuple
from copy import deepcopy
from typing import List

import gym
import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

from agents import ContinuousActorCriticAgent
from util import evaluate, plot_rewards, render_interaction

# For reproducibility
torch.manual_seed(24)
random.seed(24)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


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
    pi_optimizer = torch.optim.Adam(agent.pi.parameters(), lr=alpha)
    q_optimizer = torch.optim.Adam(agent.q.parameters(), lr=alpha)
    target_pi = deepcopy(agent.pi).to(device)
    target_q = deepcopy(agent.q).to(device)
    experience_buf = Buffer(buffer_capacity)
    total_rewards = []

    for _ in tqdm(range(epochs), disable=not verbose):
        s = torch.from_numpy(env.reset()).float()
        done = False
        reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            # Collect and save experience from the environment
            # Add Gaussian noise to the action for exploration
            a = agent.act(s) + torch.normal(mean=0.0, std=act_noise, size=(1,))
            s_prime, r, done, _ = env.step(a)
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
        description="Execute Deep Deterministic Policy Gradients against Pendulum-v0 environment"
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
    env = gym.make("Pendulum-v0")
    # For reproducibility
    env.seed(24)

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

    plot_rewards(rewards, title="DDPG on Pendulum-v0", output_dir="ddpg", filename="Pendulum-v0")

    print("Evaluating agent")
    evaluate(env, agent, args.eval_episodes, args.verbose)

    if args.save_gif:
        print("Rendering interaction")
        render_interaction(env, agent, output_dir="ddpg", filename="Pendulum-v0")
