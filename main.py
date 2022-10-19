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
    #we will make random pixel images
    #labels will be the true image and the syle and content representation
    #x is here for no reason just a placeholder
    x: np.ndarray = field(init=False)

    def __post_init__(self):
        print("hi")

#GET CONTENT REPRESENTATION

#GET STYLE REPRESENTATION

@memory.cache()
def Model():
    #model will take random white noise and make the true image
    #IMPLEMENT VGG NETWORK
    #USE FEATURE SPACE OF 16 CONV AND 5 POOLING LAYERS OF 19 LAYER VGG
    #REPLACE MAX POOLING WITH AVERAGE POOLING
    print("hi")


def main():
    #parse flags before use
    FLAGS(sys.argv)
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size

if __name__ == "__main__":
    main()
