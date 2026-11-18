"""Tabular Q-learning agent with epsilon-greedy exploration."""
import numpy as np


class QLearningAgent:
    def __init__(self, bins, n_actions, alpha=0.1, gamma=0.99, seed=0):
        self.Q = np.zeros((bins, bins, n_actions))   # Q-table: (position, velocity, action)
        self.alpha = alpha                            # learning rate (can be changed during training)
        self.gamma = gamma                            # discount factor
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def greedy(self, state):
        """Best action; if several actions tie (e.g. unvisited state, all Q=0),
        pick one of them at random instead of always action 0."""
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def act(self, state, epsilon):
        """Epsilon-greedy: random action with prob. epsilon, else the greedy action."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(state)

    def update(self, s, a, r, s_next, terminated):
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        At the goal (terminated) there is no future, so the target is just r."""
        target = r if terminated else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])