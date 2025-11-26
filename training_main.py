# training_main.py
import os
import numpy as np
from collections import deque

from model import TrainModel
from memory import ReplayMemory
from training_simulation import TrafficEnv
from visualization import plot_training_curves
from utils import ensure_dir


def train_dqn(
    episodes: int = 200,
    max_steps: int = 200,
    batch_size: int = 64,
    memory_capacity: int = 20_000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    tau: float = 0.01,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 150,
    model_dir: str = "models",
):
    env = TrafficEnv(max_episode_steps=max_steps)
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = TrainModel(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=gamma,
        lr=lr,
        tau=tau,
    )
    memory = ReplayMemory(capacity=memory_capacity)

    ensure_dir(model_dir)
    model_path = os.path.join(model_dir, "traffic_dqn_weights.h5")

    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / max(epsilon_decay_episodes, 1)

    episode_rewards = []
    episode_waiting = []

    reward_window = deque(maxlen=20)

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        total_wait = 0.0

        for step in range(max_steps):
            action = agent.act(state, epsilon)
            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            total_wait += info["total_waiting"]

            if memory.can_sample(batch_size):
                batch = memory.sample(batch_size)
                loss = agent.train_on_batch(*batch)
            else:
                loss = None

            if done:
                break

        # Epsilon decay
        if ep <= epsilon_decay_episodes:
            epsilon = max(epsilon_end, epsilon - epsilon_decay)

        episode_rewards.append(total_reward)
        episode_waiting.append(total_wait / max_steps)
        reward_window.append(total_reward)

        avg_last = np.mean(reward_window)
        print(
            f"Episode {ep:4d}/{episodes} | "
            f"Reward: {total_reward:8.2f} | "
            f"Avg last 20: {avg_last:8.2f} | "
            f"Epsilon: {epsilon:5.2f}"
        )

    # Save model
    agent.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Plot curves
    plot_training_curves(episode_rewards, episode_waiting, out_dir="plots")


if __name__ == "__main__":
    train_dqn()
