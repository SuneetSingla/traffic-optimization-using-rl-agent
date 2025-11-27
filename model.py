# model.py  (TensorFlow-free, Streamlit-deployable)
import numpy as np

class DQN_Agent:
    """
    Lightweight NumPy-based DQN forward-pass
    Works for inference without TensorFlow.
    """

    def __init__(self, model_path="models/dqn_weights.npz"):
        data = np.load(model_path)

        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]

        self.action_dim = self.W2.shape[1]

    def forward(self, state):
        """Forward pass for Q-value prediction"""
        x = np.dot(state, self.W1) + self.b1
        x = np.maximum(x, 0)  # ReLU
        q = np.dot(x, self.W2) + self.b2
        return q

    def act(self, state, epsilon=0.05):
        """
        Epsilon-greedy policy for live environment simulation.
        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_dim)  # random action
        q_values = self.forward(np.array(state, dtype=np.float32))
        return int(np.argmax(q_values))
