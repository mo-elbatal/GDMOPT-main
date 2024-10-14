import platform

# MacOS doesn't work with the default `tkinter` backend
if platform.system() == "Darwin":
    import matplotlib

    matplotlib.use("macosx")

# pylint: disable=wrong-import-position
from pathlib import Path
from typing import List, Tuple

import gym

# pylint: disable=ungrouped-imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from tqdm import tqdm

from agents import Agent


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
