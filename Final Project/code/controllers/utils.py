"Module defining utility functions"
import random
import numpy as np
import tensorflow as tf
import cloudpickle


def set_seeds(seed_num: int):
    "set the seeds for the modules random, tensorflow and numpy."
    random.seed(seed_num)
    np.random.seed(seed_num)
    tf.random.set_seed(seed_num)


def save(file, path):
    "Save a class."
    with open(path, "wb") as f:
        cloudpickle.dump(file, f)


def load(path):
    "Read a class"
    with open(path, "rb") as f:
        return cloudpickle.load(f)
