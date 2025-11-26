# gpucheck.py
import tensorflow as tf

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("GPUs available:")
        for g in gpus:
            print("  ", g)
    else:
        print("No GPU detected. TensorFlow will use CPU.")
