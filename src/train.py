"""Train a tabular Q-learning agent on MountainCar-v0, evaluate it, save curve + results.

Usage examples:
    python src/train.py
    python src/train.py --episodes 20000
    python src/train.py --episodes 5000 --eps-start 0.1
"""
import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent import QLearningAgent
from discretizer import Discretizer


def linear_epsilon(ep, total, start=1.0, end=0.01, decay_frac=0.5):
    """Epsilon goes linearly from `start` to `end` over the first `decay_frac` of training."""
    return max(end, start - (start - end) * ep / (total * decay_frac))


def train(bins, episodes, alpha, gamma, seed, eps_start=1.0, eps_end=0.01, decay_frac=0.5):
    """Run Q-learning for `episodes` episodes. Returns env, discretizer, agent, rewards list."""
    env = gym.make("MountainCar-v0")
    disc = Discretizer(env.observation_space.low, env.observation_space.high, bins)
    agent = QLearningAgent(bins, env.action_space.n, alpha, gamma, seed)

    returns = []
    for ep in range(episodes):
        eps = linear_epsilon(ep, episodes, eps_start, eps_end, decay_frac)
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        s, total, done = disc(obs), 0.0, False
        while not done:
            a = agent.act(s, eps)
            obs, r, terminated, truncated, _ = env.step(a)
            s_next = disc(obs)
            agent.update(s, a, r, s_next, terminated)
            s, total, done = s_next, total + r, terminated or truncated
        returns.append(total)
        if (ep + 1) % 500 == 0:
            print(f"Episode {ep + 1:5d} | eps={eps:.3f} | avg reward (last 100) = "
                  f"{np.mean(returns[-100:]):.1f}")
    return env, disc, agent, returns


def evaluate(env, disc, agent, n=100, seed=0):
    """Greedy policy (epsilon=0) over n episodes. Gym 'solved' = mean >= -110."""
    scores = []
    for k in range(n):
        obs, _ = env.reset(seed=10_000_000 + seed * 1000 + k)
        s, total, done = disc(obs), 0.0, False
        while not done:
            obs, r, terminated, truncated, _ = env.step(agent.act(s, 0.0))
            s, total, done = disc(obs), total + r, terminated or truncated
        scores.append(total)
    return np.mean(scores), np.std(scores)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--decay-frac", type=float, default=0.5)
    args = p.parse_args()

    env, disc, agent, returns = train(args.bins, args.episodes, args.alpha, args.gamma,
                                      args.seed, args.eps_start, args.eps_end, args.decay_frac)
    mean, std = evaluate(env, disc, agent, seed=args.seed)
    first_goal = next((i for i, r in enumerate(returns) if r > -200), None)
    print(f"\nFirst episode reaching the goal: {first_goal}")
    print(f"Greedy evaluation over 100 episodes: {mean:.1f} ± {std:.1f} "
          f"({'SOLVED' if mean >= -110 else 'not solved'}, threshold = -110)")

    # save results
    os.makedirs("results", exist_ok=True)
    tag = (f"bins{args.bins}_a{args.alpha}_g{args.gamma}_e{args.eps_start}"
           f"_ep{args.episodes}_s{args.seed}")
    pd.DataFrame({"episode": range(len(returns)), "reward": returns}).to_csv(
        f"results/returns_{tag}.csv", index=False)

    smooth = pd.Series(returns).rolling(100).mean()
    plt.figure(figsize=(9, 4.5))
    plt.plot(returns, alpha=0.25, label="Reward per episode")
    plt.plot(smooth, linewidth=2, label="Moving average (100 episodes)")
    plt.axhline(-110, color="red", linestyle="--", label="Solved threshold (-110)")
    plt.xlabel("Episode"); plt.ylabel("Cumulative reward")
    plt.title(f"Tabular Q-learning on MountainCar-v0 ({tag})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"results/curve_{tag}.png", dpi=150)
    print(f"Saved results/curve_{tag}.png")