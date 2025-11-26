# testing_main.py
import os
from model import TrainModel
from testing_simulation import evaluate_policy
from training_simulation import TrafficEnv


def main():
    env = TrafficEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = TrainModel(state_dim, action_dim)
    model_path = os.path.join("models", "traffic_dqn_weights.h5")

    if not os.path.exists(model_path):
        print("Trained model not found. First run: python training_main.py")
        return

    agent.load(model_path, state_dim, action_dim)
    print(f"Loaded trained model from {model_path}")

    rewards, waiting = evaluate_policy(agent, episodes=10, max_steps=200)
    print("\n=== TEST SUMMARY ===")
    print(f"Average reward: {rewards.mean():.2f}")
    print(f"Average waiting: {waiting.mean():.2f}")


if __name__ == "__main__":
    main()
