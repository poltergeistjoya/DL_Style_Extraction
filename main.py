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

#@dataclass
#class Data:
#    #we will make random pixel images
#    #labels will be the true image and the syle and content representation
#    rng: InitVar[np.random.Generator]
#    cont: tf.Variable(trainable = True)
#
#    def __post_init__(self):
#        print("hi")
#        self.cont = rng.uniform(0,255,size = [1,224,224,3])

#make trainable content and style
class Data(tf.Module):
    def __init__(self, rng):
        #content
        self.cont = tf.Variable(rng.normal(loc = 0.0,scale =255.0,size= [1,224,224,3]), trainable = True)
        #style
        self.style = tf.Variable(rng.normal(loc = 0.0,scale =255.0,size= [1,224,224,3]), trainable = True)


#GET CONTENT REPRESENTATION

#GET STYLE REPRESENTATION

#crop and resize image to 224 by 224
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
    return img_rz

#convert img object to VGG accepted format
def img_to_VGG(img):
    np_img = np.array(img)
    #add fourth dimension, # of images
    VGG_img = np.expand_dims(np_img, axis = 0)
    return VGG_img

#done for chosen content layer, paper says conv5_2
def content_loss(gen_cont, true_cont):
    return K.sum(K.square(target - base_content))

#style loss function

@memory.cache()
def vgg_16():
    #IMPLEMENT VGG NETWORK
    #USE FEATURE SPACE OF 16 CONV AND 5 POOLING LAYERS OF 19 LAYER VGG
    #REPLACE MAX POOLING WITH AVERAGE POOLING
    model = VGG16(include_top=False, weights = "imagenet")

    #opt = optimizers.Adam(lr = 0.001)
    #model.compile(optimizer = opt, loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])
    print(model.summary())
    return model

def main():
    #parse flags before use
    FLAGS(sys.argv)
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size

    #random number gen
    np_rng = np.random.default_rng(31415)

    model = vgg_16()
    #for layer in model.layers:
    #    print(layer)
    path = "starry_night.jpeg"
    true_img = img_resize(path)
    true = img_to_VGG(true_img)
    print(true.shape)

    #pass through true image and save output of conv layers
    content_layers = ['block5_conv2']

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    print(outputs_dict['block5_conv2'])


    data = Data(np_rng)
    #plt.imshow(data.cont, interpolation= 'nearest')
    #plt.show()
    #plt.savefig('rand.png')

    #pass through white noise image(tf.variable)and


if __name__ == "__main__":
    main()
