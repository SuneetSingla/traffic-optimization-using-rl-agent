import numpy as np
import tensorflow as tf

# Correct architecture (based on weight shapes)
state_dim = 5   # input features
action_dim = 2  # number of actions used in training

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(state_dim,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(action_dim)
])

model.load_weights("models/traffic_dqn_weights.h5")

weights = model.get_weights()
np.savez("models/dqn_weights.npz", *weights)

print("NPZ FILE CREATED → models/dqn_weights.npz")
