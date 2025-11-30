import numpy as np

class DQN_Agent:
    def __init__(self, npz_path, state_dim=5, action_dim=2):
        weights = np.load(npz_path)
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = weights.values()
        self.action_dim = action_dim

    def forward(self, x):
        x = np.maximum(0, x @ self.w1 + self.b1)
        x = np.maximum(0, x @ self.w2 + self.b2)
        return x @ self.w3 + self.b3

    def act(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        q = self.forward(np.array(state, dtype=np.float32))
        return int(np.argmax(q))