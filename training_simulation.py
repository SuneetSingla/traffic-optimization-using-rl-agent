# training_simulation.py
import numpy as np
from generator import TrafficGenerator


class TrafficEnv:
    """
    Simple single-intersection environment for DQN, no SUMO needed.

    - 4 approaches: N, E, S, W
    - State: [queue_N, queue_E, queue_S, queue_W, current_phase]
      where current_phase in {0, 1}
        0 = NS green, EW red
        1 = EW green, NS red

    - Action:
        0 -> keep current phase
        1 -> switch phase

    - Reward:
        negative total waiting cars at current step
        (so fewer cars in queue is better: higher reward)
    """

    def __init__(
        self,
        max_episode_steps: int = 200,
        green_throughput_per_step: int = 3,
        arrival_rate: float = 0.4,
        switch_penalty: float = 1.0,
    ):
        self.max_steps = max_episode_steps
        self.green_throughput = green_throughput_per_step
        self.switch_penalty = switch_penalty

        self.generator = TrafficGenerator(arrival_rate_per_lane=arrival_rate)

        self.step_count = 0
        self.queue = np.zeros(4, dtype=np.float32)  # [N, E, S, W]
        self.phase = 0  # 0: NS green, 1: EW green

    @property
    def state_dim(self):
        # 4 queues + 1 phase
        return 5

    @property
    def action_dim(self):
        # keep / switch
        return 2

    def reset(self):
        self.step_count = 0
        self.queue[:] = 0
        self.phase = 0
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.queue, np.array([self.phase], dtype=np.float32)])

    def step(self, action: int):
        """
        Take an action and update queues.
        """
        self.step_count += 1

        # Phase switching
        switch = (action == 1)
        if switch:
            self.phase = 1 - self.phase

        # Cars arrive
        arrivals = self.generator.sample_arrivals()
        self.queue += arrivals

        # Discharge cars on green approaches
        if self.phase == 0:  # NS green
            # N and S discharge
            for i in [0, 2]:
                served = min(self.green_throughput, int(self.queue[i]))
                self.queue[i] -= served
        else:  # EW green
            for i in [1, 3]:
                served = min(self.green_throughput, int(self.queue[i]))
                self.queue[i] -= served

        # Compute reward
        total_waiting = float(np.sum(self.queue))
        reward = -total_waiting
        if switch:
            reward -= self.switch_penalty  # discourage unnecessary switching

        done = self.step_count >= self.max_steps
        next_state = self._get_state()

        info = {"total_waiting": total_waiting}
        return next_state, reward, done, info
