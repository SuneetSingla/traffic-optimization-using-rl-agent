# testing_simulation.py
import numpy as np
from training_simulation import TrafficEnv


def evaluate_policy(agent, episodes: int = 10, max_steps: int = 200):
    env = TrafficEnv(max_episode_steps=max_steps)
    rewards = []
    waiting = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        total_wait = 0.0

        for step in range(max_steps):
            # greedy (epsilon = 0)
            action = agent.act(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            state = next_state

            total_reward += reward
            total_wait += info["total_waiting"]
            if done:
                break

        rewards.append(total_reward)
        waiting.append(total_wait / max_steps)
        print(
            f"[TEST] Episode {ep+1}/{episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Avg waiting: {waiting[-1]:.2f}"
        )

    return np.array(rewards), np.array(waiting)
