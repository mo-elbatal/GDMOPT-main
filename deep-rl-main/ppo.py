import argparse
from collections import namedtuple
from typing import List, Tuple

import gym
import torch
import torch.optim as optim
from tqdm import tqdm

from agents import DiscreteActorCriticAgent
from util import evaluate, plot_rewards, render_interaction

# For reproducibility
torch.manual_seed(24)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Episode = namedtuple("Episode", ["states", "actions", "rewards"])
Data = namedtuple("Data", ["values", "log_probs", "returns", "advantages"])


# pylint: disable=too-many-locals
def ppo(
    env: gym.Env,
    agent: DiscreteActorCriticAgent,
    epochs: int,
    num_episodes: int,
    max_steps: int,
    pi_epochs: int,
    alpha: float,
    gamma: float,
    ratio_epsilon: float,
    verbose: bool,
) -> List[float]:
    """Trains an agent using proximal policy optimization algorithm

    :param env: The environment to train the agent in
    :type env: gym.Env
    :param agent: The agent to train
    :type agent: DiscreteActorCriticAgent
    :param epochs: The number of epochs to train the agent for
    :type epochs: int
    :param num_episodes: The number of episodes to sample per epoch
    :type num_episodes: int
    :param max_steps: The max number of steps per episode
    :type max_steps: int
    :param pi_epochs: The number of epochs to use for updating the policy approximator
    :type pi_epochs: int
    :param alpha: The learning rate
    :type alpha: float
    :param gamma: The discount factor
    :type gamma: float
    :param ratio_epsilon: The value of epsilon for clipping the ratio
    :type ratio_epsilon: float
    :param verbose: Whether to run in verbose mode or not
    :type verbose: bool
    :return: The total reward per episode
    :rtype: List[float]
    """
    pi_optimizer = optim.Adam(agent.pi.parameters(), lr=alpha)
    v_optimizer = optim.Adam(agent.v.parameters(), lr=alpha)
    total_rewards = []

    for _ in tqdm(range(epochs), disable=not verbose):
        # Collect and process experience from the environment
        episodes, rewards = _sample_episodes(env, agent, num_episodes, max_steps)
        # This first processing generates data for the policy before the update (pi_old in the paper)
        data = _process_episodes(episodes, agent, gamma)
        total_rewards.extend(rewards)

        states = torch.cat([e.states for e in episodes], dim=0)
        actions = torch.cat([e.actions for e in episodes], dim=0)
        # We need to detach the old log probs from the computational graph because we are going to
        # compute the gradient with respect to the _current_ parameters, i.e. the ones that will
        # generate new log probs below
        log_probs_old = data.log_probs.detach()

        # Update the policy function for `pi_epochs`
        for _ in range(pi_epochs):
            pi_optimizer.zero_grad()

            _, log_probs = agent.step(states, actions)
            # Since we return the log probabilities, we need to exponentiate them and then divide them
            # Based on the law of exponents e^x / e^y = e(x-y)
            # We need to detach the
            ratio = torch.exp(log_probs - log_probs_old)
            clip_ratio = torch.clamp(ratio, 1 - ratio_epsilon, 1 + ratio_epsilon)

            # We use the negative loss because policy optimization is done using gradient _ascent_
            # This is because in policy gradient methods, the "loss" is a performance measure that is _maximized_
            pi_loss = -(torch.min(ratio * data.advantages, clip_ratio * data.advantages)).mean()
            pi_loss.backward()
            pi_optimizer.step()

        # Update the value function
        v_optimizer.zero_grad()
        v_loss = ((data.values - data.returns) ** 2).mean()
        v_loss.backward()
        v_optimizer.step()

    return total_rewards


def _sample_episodes(
    env: gym.Env, agent: DiscreteActorCriticAgent, num_episodes: int, max_steps: int
) -> Tuple[List[Episode], List[float]]:
    episodes = []
    rewards = []

    for _ in range(num_episodes):
        states = []
        actions = []
        rews = []

        s = torch.from_numpy(env.reset()).float()
        done = False
        steps = 0
        reward = 0.0

        while not done and steps < max_steps:
            a = agent.act(s)
            s_prime, r, done, _ = env.step(a)

            states.append(s)
            actions.append(torch.as_tensor(a))
            rews.append(torch.as_tensor(r))

            s = torch.from_numpy(s_prime).float()
            steps += 1
            reward += r

        # If the episode got truncated then bootstrap to approximate the missing returns
        if not done:
            a = agent.act(s)
            v, _ = agent.step(s, torch.as_tensor(a))
            states.append(s)
            actions.append(torch.as_tensor(a))
            rews.append(v)

        # Rewards are not converted to tensor since we compute the return by backtracking through the list
        episodes.append(Episode(torch.stack(states), torch.stack(actions), rews))
        rewards.append(reward)

    return episodes, rewards


def _process_episodes(episodes: List[Episode], agent: DiscreteActorCriticAgent, gamma: float) -> Data:
    values = []
    log_probs = []
    returns = []
    advantages = []

    for episode in episodes:
        rews = episode.rewards
        # Reverse the list so we start backpropagating the return from the last timestep
        rews.reverse()

        # Compute the return G_t:T
        g = 0
        G_t = []
        for r in rews:
            g = r + gamma * g
            G_t.append(g)

        # Reverse the list of returns so that we start from the first timestep
        G_t.reverse()
        G_t = torch.as_tensor(G_t).to(device)

        # Compute log pi(A_t|S_t) and Adv(S_t,A_t)
        v, log_prob = agent.step(episode.states, episode.actions)
        advantage = G_t - v.detach()

        values.append(v)
        log_probs.append(log_prob)
        returns.append(G_t)
        advantages.append(advantage)

    return Data(
        torch.cat(values, dim=0), torch.cat(log_probs, dim=0), torch.cat(returns, dim=0), torch.cat(advantages, dim=0)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Proximal Policy Optimization against CartPole-v1 environment")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to train")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes to sample per epoch")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--alpha", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument(
        "--pi-epochs", type=int, default=10, help="The number of epochs to use for updating the policy approximator"
    )
    parser.add_argument("--ratio-epsilon", type=float, default=0.2, help="The value of epsilon for clipping the ratio")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes to use for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode")
    parser.add_argument("--save-gif", action="store_true", help="Save a GIF of an interaction after training")
    args = parser.parse_args()

    agent = DiscreteActorCriticAgent(num_features=4, num_actions=2, device=device)
    env = gym.make("CartPole-v1")
    # For reproducibility
    env.seed(24)

    print(f"Training agent with the following args\n{args}")
    rewards = ppo(
        env,
        agent,
        epochs=args.epochs,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        pi_epochs=args.pi_epochs,
        alpha=args.alpha,
        gamma=args.gamma,
        ratio_epsilon=args.ratio_epsilon,
        verbose=args.verbose,
    )

    plot_rewards(rewards, title="PPO on CartPole-v1", output_dir="ppo", filename="CartPole-v1")

    print("Evaluating agent")
    evaluate(env, agent, args.eval_episodes, args.verbose)

    if args.save_gif:
        print("Rendering interaction")
        render_interaction(env, agent, output_dir="ppo", filename="CartPole-v1")
