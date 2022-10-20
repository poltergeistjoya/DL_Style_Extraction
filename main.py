#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

from absl import flags
from dataclasses import dataclass, field, InitVar
from joblib import Memory

from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, optimizers
from PIL import Image

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
def vgg_16(weights = 'imagenet', include_top = True):
    #IMPLEMENT VGG NETWORK
    #USE FEATURE SPACE OF 16 CONV AND 5 POOLING LAYERS OF 19 LAYER VGG
    #REPLACE MAX POOLING WITH AVERAGE POOLING
    model = models.Sequential()

    #Conv 1 block
    model.add(layers.Conv2D(input_shape = (224,224,3), filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.AveragePooling2D(pool_size =(2,2), strides = (2,2)))

    #Conv 2 block
    model.add(layers.Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.AveragePooling2D(pool_size =(2,2), strides = (2,2)))

    #Conv 3 block
    model.add(layers.Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.AveragePooling2D(pool_size =(2,2), strides = (2,2)))

    #Conv 4 block
    model.add(layers.Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.AveragePooling2D(pool_size =(2,2), strides = (2,2)))

    #Conv 5 block
    model.add(layers.Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
    model.add(layers.AveragePooling2D(pool_size =(2,2), strides = (2,2)))

    opt = optimizers.Adam(lr = 0.001)
    model.compile(optimizer = opt, loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])
    model.summary()
    return model

def main():
    #parse flags before use
    FLAGS(sys.argv)
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size

    model = vgg_16()

    #pass through true image and save output of conv layers

    #pass through white noise image and


if __name__ == "__main__":
    main()
