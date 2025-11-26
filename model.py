# model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DQN(keras.Model):
    """
    Simple fully-connected DQN network.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.d1 = layers.Dense(128, activation="relu")
        self.d2 = layers.Dense(128, activation="relu")
        self.out = layers.Dense(action_dim, activation="linear")

    def call(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = self.d1(inputs)
        x = self.d2(x)
        return self.out(x)


class TrainModel:
    """
    Wrapper around the DQN + target network + training step.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        tau: float = 0.005,
    ):
        self.gamma = gamma
        self.tau = tau

        self.online_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        # Build networks by calling once
        dummy = np.zeros((1, state_dim), dtype=np.float32)
        _ = self.online_net(dummy)
        _ = self.target_net(dummy)

        self.target_net.set_weights(self.online_net.get_weights())
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = keras.losses.Huber()

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones):
        """
        Single gradient update step.
        """
        # Compute target Q-values
        next_q_values = self.target_net(next_states)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        with tf.GradientTape() as tape:
            q_values = self.online_net(states)
            # Gather Q-values for taken actions
            actions_one_hot = tf.one_hot(actions, q_values.shape[-1])
            q_for_actions = tf.reduce_sum(q_values * actions_one_hot, axis=1)
            loss = self.loss_fn(targets, q_for_actions)

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))
        return loss

    def train_on_batch(self, states, actions, rewards, next_states, dones) -> float:
        loss = self._train_step_tf(
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(next_states, dtype=tf.float32),
            tf.convert_to_tensor(dones, dtype=tf.float32),
        )
        # Soft update of target network: θ_target = τ*θ_online + (1-τ)*θ_target
        self.soft_update()
        return float(loss.numpy())

    def soft_update(self):
        online_weights = self.online_net.get_weights()
        target_weights = self.target_net.get_weights()
        new_weights = []
        for ow, tw in zip(online_weights, target_weights):
            new_weights.append(self.tau * ow + (1.0 - self.tau) * tw)
        self.target_net.set_weights(new_weights)

    def act(self, state, epsilon: float):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() < epsilon:
            # explore
            return np.random.randint(0, self.online_net.out.units)
        q_values = self.online_net(np.expand_dims(state, axis=0))
        return int(np.argmax(q_values.numpy()[0]))

    def save(self, path: str):
        self.online_net.save_weights(path)

    def load(self, path: str, state_dim: int, action_dim: int):
        # Rebuild then load
        dummy = np.zeros((1, state_dim), dtype=np.float32)
        _ = self.online_net(dummy)
        self.online_net.load_weights(path)
        self.target_net.set_weights(self.online_net.get_weights())
