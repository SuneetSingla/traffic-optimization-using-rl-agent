import numpy as np

class TrafficEnvironment:
    """
    Simple 4-way intersection environment for DQN agent.
    State = queue lengths for [N, E, S, W]
    Actions = 0 = NS green, 1 = EW green
    Reward encourages lower waiting / queue length.
    """

    def __init__(self, arrival_rate=2, max_steps=200):
        self.arrival_rate = arrival_rate
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.queues = np.zeros(4, dtype=np.int32)  # [N,E,S,W]
        self.steps = 0
        return self._get_state()

    def step(self, action):
        """
        action = 0 → North-South is green
        action = 1 → East-West is green
        """

        # Add arriving vehicles (Poisson distribution)
        arrivals = np.random.poisson(self.arrival_rate, size=4)
        self.queues += arrivals

        # Vehicles pass depending on action
        if action == 0:   # NS Green
            self.queues[0] = max(0, self.queues[0] - np.random.randint(1, 4))
            self.queues[2] = max(0, self.queues[2] - np.random.randint(1, 4))
        else:             # EW Green
            self.queues[1] = max(0, self.queues[1] - np.random.randint(1, 4))
            self.queues[3] = max(0, self.queues[3] - np.random.randint(1, 4))

        # Reward → negative waiting
        reward = -np.sum(self.queues)

        self.steps += 1
        done = self.steps >= self.max_steps

        return self._get_state(), reward, done, {}

    def get_state(self):
        """
        Returns a numeric state vector representing current traffic.
        State = [cars_north, cars_east, cars_south, cars_west, current_phase]
        """
        return np.array([
            len(self.queue_N),
            len(self.queue_E),
            len(self.queue_S),
            len(self.queue_W),
            self.current_phase
        ], dtype=np.float32)


    def render(self):
        print(f"Queues [N,E,S,W] = {self.queues}")
