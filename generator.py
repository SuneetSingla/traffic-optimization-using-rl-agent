# generator.py
import numpy as np

class TrafficGenerator:
    """
    Traffic generator with adjustable arrival rate.
    arrival_rate_per_lane controls how many cars (avg) enter per lane per step.
    """

    def __init__(self, arrival_rate_per_lane: float = 4.0, seed: int = 42):
        self.arrival_rate = arrival_rate_per_lane
        self.rng = np.random.default_rng(seed)

    def set_rate(self, new_rate: float):
        """Update arrival rate live during UI simulation."""
        self.arrival_rate = new_rate

    def sample_arrivals(self):
        """Poisson arrivals based on live adjustable rate."""
        return self.rng.poisson(lam=self.arrival_rate, size=4).astype(int)
