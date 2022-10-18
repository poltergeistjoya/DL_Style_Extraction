#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

from absl import flags
from dataclasses import dataclass, field, InitVar
from joblib import Memory

memory = Memory(".cache")

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 1024, "Number of samples in a batch")
flags.DEFINE_integer("epochs", 5, "Number of epochs")

@dataclass
class Data:

    #x is here for no reason just a placeholder
    x: np.ndarray = field(init=False)

    def __post_init__(self):
        print("hi")

@memory.cache()
def Model():
    print("hi")


def main():
    #parse flags before use
    FLAGS(sys.argv)
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size

if __name__ == "__main__":
    main()
