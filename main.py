#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

from absl import flags
from dataclasses import dataclass, field, InitVar
from joblib import Memory

from tensorflow import keras
from keras import optimizers
from keras import backend as K
from keras.applications.vgg16 import VGG16
#from tensorflow.keras import layers, models, regularizers, optimizers
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

#resize image to 224 by 224
#crop
def img_resize(img_path):
    baseheight = 224
    img = Image.open(img_path)
    width = img.size[0]
    height = img.size[1]

    crop_by = width if width < height else height
    img_crop = img.crop((0,0,crop_by,crop_by))
    hpercent = (baseheight / float(crop_by))


    width = int((float(img_crop.size[0]) * float(hpercent)))
    img_rz = img_crop.resize((width, baseheight), Image.ANTIALIAS)
    img_rz.save('resizedimage.jpg')
    print(img_rz.size)
    return img_rz

@memory.cache()
def vgg_16():
    #IMPLEMENT VGG NETWORK
    #USE FEATURE SPACE OF 16 CONV AND 5 POOLING LAYERS OF 19 LAYER VGG
    #REPLACE MAX POOLING WITH AVERAGE POOLING
    model = VGG16(include_top=False, weights = "imagenet")

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
    #for layer in model.layers:
    #    print(layer)
    path = "starry_night.jpeg"
    true_img = img_resize(path)

    #pass through true image and save output of conv layers

    #pass through white noise image(tf.variable)and


if __name__ == "__main__":
    main()
