# visualization.py
import os
import matplotlib.pyplot as plt

from utils import ensure_dir


def plot_training_curves(rewards, avg_waiting, out_dir="plots"):
    ensure_dir(out_dir)

    # Rewards
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward per Episode")
    plt.tight_layout()
    reward_path = os.path.join(out_dir, "rewards.png")
    plt.savefig(reward_path)
    plt.close()

    # Average waiting
    plt.figure()
    plt.plot(avg_waiting)
    plt.xlabel("Episode")
    plt.ylabel("Avg Waiting (cars)")
    plt.title("Average Waiting per Episode")
    plt.tight_layout()
    wait_path = os.path.join(out_dir, "avg_waiting.png")
    plt.savefig(wait_path)
    plt.close()

    print(f"Plots saved to {out_dir}/")
