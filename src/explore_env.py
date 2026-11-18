"""Explore MountainCar-v0: spaces, limits, starting states, random vs heuristic agent."""
import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0")

print("=== Spaces ===")
print("Observation low :", env.observation_space.low)
print("Observation high:", env.observation_space.high)
print("Actions         :", env.action_space)
print("Reward threshold:", env.spec.reward_threshold)
print("Max steps       :", env.spec.max_episode_steps)

print("\n=== Starting positions (10 resets) ===")
starts = [env.reset(seed=i)[0][0] for i in range(10)]
print(np.round(starts, 3))


def run(policy, episodes=10):
    """Run a policy, return list of (total_reward, reached_goal, max_position)."""
    out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total, done, max_pos = 0, False, obs[0]
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            total += r
            max_pos = max(max_pos, obs[0])
            done = term or trunc
        out.append((total, term, round(float(max_pos), 3)))
    return out


print("\n=== Random agent ===")
for res in run(lambda obs: env.action_space.sample()):
    print(res)

print("\n=== Heuristic agent (push in direction of velocity) ===")
heur = run(lambda obs: 2 if obs[1] >= 0 else 0)
for res in heur:
    print(res)
print("Average reward:", np.mean([r[0] for r in heur]))

env.close()