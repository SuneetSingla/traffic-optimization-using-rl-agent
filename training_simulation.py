# # training_simulation.py
# import numpy as np
# from generator import TrafficGenerator


# class TrafficEnv:

#     def __init__(
#         self,
#         max_episode_steps: int = 200,
#         green_throughput_per_step: int = 3,
#         arrival_rate: float = 0.4,
#         switch_penalty: float = 1.0,
#         initial_cars=None  # 🔥 NEW
#     ):
#         self.max_steps = max_episode_steps
#         self.green_throughput = green_throughput_per_step
#         self.switch_penalty = switch_penalty
#         self.generator = TrafficGenerator(arrival_rate_per_lane=arrival_rate)

#         # QUEUES -------------------------
#         if initial_cars is not None:
#             self.queue = np.array(initial_cars, dtype=np.float32)   # 🔥 user-defined start
#         else:
#             self.queue = np.zeros(4, dtype=np.float32)

#         self.phase = 0
#         self.step_count = 0


#     @property
#     def state_dim(self): return 5  
#     @property
#     def action_dim(self): return 2


#     def reset(self, initial_cars=None):
#         """
#         Reset simulation (now supports custom starting queues)
#         """
#         self.step_count = 0

#         if initial_cars is not None:
#             self.queue = np.array(initial_cars, dtype=np.float32)
#         # ✔ otherwise keep last controlled UI queue values

#         self.phase = 0
#         return self._get_state()


#     def _get_state(self):
#         return np.concatenate([self.queue, [self.phase]], dtype=np.float32)


#     def step(self, action:int):
#         self.step_count += 1

#         # Switching phase (0↔1)
#         if action == 1:
#             self.phase = 1 - self.phase

#         # Arrival of cars
#         arrivals = self.generator.sample_arrivals()
#         self.queue += arrivals

#         # Serve cars
#         if self.phase == 0:             # NS green
#             for i in [0, 2]:            # queue_N, queue_S
#                 self.queue[i] = max(self.queue[i] - self.green_throughput, 0)
#         else:                           # EW green
#             for i in [1, 3]:            # queue_E, queue_W
#                 self.queue[i] = max(self.queue[i] - self.green_throughput, 0)

#         # Reward = lower queue → better
#         total_waiting = float(self.queue.sum())
#         reward = -total_waiting
#         if action == 1: reward -= self.switch_penalty

#         done = self.step_count >= self.max_steps

#         return self._get_state(), reward, done, {"total_waiting":total_waiting}





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
        initial_cars=None,
    ):
        self.max_steps = max_episode_steps
        self.green_throughput = green_throughput_per_step
        self.switch_penalty = switch_penalty

        self.generator = TrafficGenerator(arrival_rate_per_lane=arrival_rate)

        self.step_count = 0

        # queue: [N, E, S, W]
        if initial_cars is not None:
            self.queue = np.array(initial_cars, dtype=np.float32)
        else:
            self.queue = np.zeros(4, dtype=np.float32)

        # 0: NS green, 1: EW green
        self.phase = 0

    # ------------- Spaces -------------
    @property
    def state_dim(self):
        # 4 queues + 1 phase
        return 5

    @property
    def action_dim(self):
        # keep / switch
        return 2

    # ------------- Core MDP methods -------------
    def reset(self, initial_cars=None):
        """
        Reset simulation (supports optional custom queues).
        """
        self.step_count = 0

        if initial_cars is not None:
            self.queue = np.array(initial_cars, dtype=np.float32)
        else:
            # default: start empty if nothing provided
            # (for training scripts that call reset() with no args)
            self.queue[:] = 0.0

        self.phase = 0
        return self._get_state()

    def _get_state(self):
        # state = [queue_N, queue_E, queue_S, queue_W, phase]
        return np.concatenate([self.queue, np.array([self.phase], dtype=np.float32)])

    def step(self, action: int):
        """
        Take an action and update queues.

        action:
            0 = keep current phase
            1 = switch to other phase
        """
        self.step_count += 1

        # Phase switching
        switch = (action == 1)
        if switch:
            self.phase = 1 - self.phase

        # Cars arrive
        arrivals = self.generator.sample_arrivals()   # shape (4,)
        self.queue += arrivals

        # Discharge cars on green approaches
        if self.phase == 0:  # NS green
            for i in [0, 2]:  # N and S
                served = min(self.green_throughput, int(self.queue[i]))
                self.queue[i] -= served
        else:  # EW green
            for i in [1, 3]:  # E and W
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
