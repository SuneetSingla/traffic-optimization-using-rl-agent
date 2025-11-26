# utils.py
import os


def ensure_dir(path: str):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
