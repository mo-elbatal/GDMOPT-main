import argparse
import random
from collections import namedtuple
from typing import List

import gym
import torch
import torch.optim as optim
from tqdm import tqdm

from agents import QAgent
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
def dqn(
    env: gym.Env,
    agent: QAgent,
    epochs: int,
    max_steps: int,
    buffer_capacity: int,
    batch_size: int,
    alpha: float,
    gamma: float,
    verbose: bool,
) -> List[float]:
    """Trains an agent using Deep Q-network algorithm

    :param env: The environment to train the agent in
    :type env: gym.Env
    :param agent: The agent to train
    :type agent: QAgent
    :param epochs: The number of epochs to train the agent for
    :type epochs: int
    :param max_steps: The max number of steps per episode
    :type max_steps: int
    :param buffer_capacity: Max capacity of the experience replay buffer
    :type buffer_capacity: int
    :param batch_size: Batch size to use of experiences from the buffer
    :type batch_size: int
    :param alpha: The learning rate
    :type alpha: float
    :param gamma: The discount factor
    :type gamma: float
    :param verbose: Whether to run in verbose mode or not
    :type verbose: bool
    :return: The total reward per episode
    :rtype: List[float]
    """
    q_optimizer = optim.Adam(agent.q.parameters(), lr=alpha)
    experience_buf = Buffer(buffer_capacity)
    total_rewards = []

    for _ in tqdm(range(epochs), disable=not verbose):
        s = torch.from_numpy(env.reset()).float()
        done = False
        reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            # Collect and save experience from the environment
            a = agent.act(s)
            s_prime, r, done, _ = env.step(a)
            s_prime = torch.from_numpy(s_prime).float()

            reward += r
            experience_buf.save(Experience(s, a, r, s_prime, done))

            # Learn from previous experiences
            experiences = experience_buf.get(batch_size)
            loss = 0.0

            states = torch.stack([e.state for e in experiences])
            actions = [e.action for e in experiences]
            rewards = [e.reward for e in experiences]
            next_states = torch.stack([e.next_state for e in experiences])
            dones = [e.done for e in experiences]

            q_values = agent.step(states)
            next_qvalues = agent.step(next_states)
            # Keep a copy of the current Q-values to be used for the TD targets
            td_targets = q_values.clone()

            # Compute TD targets
            for index in range(batch_size):
                # Terminal states do not have a future value
                next_qvalue = 0.0
                if not dones[index]:
                    next_qvalue = torch.max(next_qvalues[index])

                # Only update the target value for the action that was taken
                td_targets[index][actions[index]] = rewards[index] + gamma * next_qvalue

            # Compute TD error and loss (MSE)
            loss = (td_targets - q_values) ** 2
            loss = loss.mean()

            # Update the value function
            q_optimizer.zero_grad()
            loss.sum().backward()
            q_optimizer.step()

            s = s_prime
            steps += 1

        total_rewards.append(reward)

    return total_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Deep Q-learning against CartPole-v1 environment")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to train")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--buf-capacity", type=int, default=50000, help="Max capacity of the experience replay buffer")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size to use of experiences from the buffer")
    parser.add_argument("--alpha", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--initial-epsilon", type=float, default=1.0, help="Initial exploration factor")
    parser.add_argument("--final-epsilon", type=float, default=0.1, help="Final exploration factor")
    parser.add_argument("--epsilon-decay", type=float, default=10.0, help="Decay for exploration factor")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes to use for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode")
    parser.add_argument("--save-gif", action="store_true", help="Save a GIF of an interaction after training")
    args = parser.parse_args()

    agent = QAgent(
        num_features=4,
        num_actions=2,
        initial_epsilon=args.initial_epsilon,
        final_epsilon=args.final_epsilon,
        epsilon_decay=args.epsilon_decay,
        device=device,
    )
    env = gym.make("CartPole-v1")
    # For reproducibility
    env.seed(24)

    print(f"Training agent with the following args\n{args}")
    rewards = dqn(
        env,
        agent,
        epochs=args.epochs,
        max_steps=args.max_steps,
        buffer_capacity=args.buf_capacity,
        batch_size=args.batch_size,
        alpha=args.alpha,
        gamma=args.gamma,
        verbose=args.verbose,
    )

    plot_rewards(rewards, title="DQN on CartPole-v1", output_dir="dqn", filename="CartPole-v1")

    # For evaluation purposes, we now want the agent to be purely greedy
    agent.explore(False)

    print("Evaluating agent")
    evaluate(env, agent, args.eval_episodes, args.verbose)

    if args.save_gif:
        print("Rendering interaction")
        render_interaction(env, agent, output_dir="dqn", filename="CartPole-v1")
