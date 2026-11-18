"""Turn MountainCar's continuous state (position, velocity) into discrete grid indices."""
import numpy as np


class Discretizer:
    """Uniform grid: each state dimension is split into `bins` equal intervals."""

    def __init__(self, low, high, bins):
        self.bins = bins
        # inner edges only: np.digitize then returns indices 0 .. bins-1
        self.edges = [np.linspace(low[i], high[i], bins + 1)[1:-1] for i in range(len(low))]

    def __call__(self, obs):
        """Continuous observation -> tuple of bin indices, e.g. (7, 10)."""
        return tuple(int(np.digitize(obs[i], self.edges[i])) for i in range(len(obs)))