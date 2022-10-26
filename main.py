#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

from absl import flags
from dataclasses import dataclass, field, InitVar
from joblib import Memory
from tqdm import trange

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
flags.DEFINE_float("lr", .1, "Learning rate for ADAM")
flags.DEFINE_integer("num_iters", 100000, "number of iterations for ADAM")

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
#TRY NORMALIZING
class Data(tf.Module):
    def __init__(self, rng):
        #content
        mu = np.float32(0)
        sigmas = np.float32(255)
        self.cont = tf.Variable(rng.normal(loc = 0.0,scale =1.0,size= [1,224,224,3]), trainable = True, dtype=tf.float32)

        #style
        #self.style = tf.Variable(rng.normal(loc = 0.0,scale =255.0,size= [1,224,224,3]), trainable = True)



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

    #normalize image
    img_arr = np.array(img_rz)
    img_rz_nm = img_arr.astype('float32')/255.0
    return img_rz_nm

#convert img object to VGG accepted format
def img_to_VGG(img):
    np_img = np.array(img)
    #add fourth dimension, # of images
    VGG_img = np.expand_dims(np_img, axis = 0)
    return VGG_img

#done for chosen content layer, paper says conv5_2
def content_loss(gen_cont, true_cont):
    return 0.5*K.sum(K.square(true_cont - gen_cont))

#style loss function

@memory.cache()
def vgg_16():
    #IMPLEMENT VGG NETWORK
    #USE FEATURE SPACE OF 16 CONV AND 5 POOLING LAYERS OF 19 LAYER VGG
    #REPLACE MAX POOLING WITH AVERAGE POOLING
    model = VGG16(include_top=False, weights = "imagenet")

    print(model.summary())
    return model

def content_extract(model, data, content_layers):
    feature_extractor = keras.Model(inputs = model.input, outputs = model.get_layer(content_layers).output)
    features = feature_extractor(data)
    return features

def main():

    #parse flags before use
    FLAGS(sys.argv)
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    lr = FLAGS.lr
    iters = FLAGS.num_iters

    #random number gen
    np_rng = np.random.default_rng(31415)

    #true image to get content and style
    path = "starry_night.jpeg"
    true_img = img_resize(path)
    true = img_to_VGG(true_img)
    #print(true.shape)

    #randomly generated white noise
    data = Data(np_rng)

    #initialize vgg model
    model = vgg_16()

    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #        initial_learning_rate = 1e-1,
    #        decay_steps=1000,
    #        decay_rate=0.9)

    boundaries = [300, 1000, 5000, 20000, 50000, 75000]
    values=[.1, .08, .05, .025, .01, .001, .0001]

    lr_schedule=keras.optimizers.schedules.PiecewiseConstantDecay(boundaries,values)

    optimizer = keras.optimizers.Adam(learning_rate = lr_schedule)

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    #content layers
    content_layers = 'block5_conv2'

    #style layers

    #get content features
    true_cont_feat=content_extract(model, true, content_layers)


    gen_cont = np.squeeze(data.cont)
    plt.imshow(gen_cont, interpolation='nearest')
    plt.show()
    plt.savefig('rand.png')

    bar = trange(iters)
    for i in bar:
        with tf.GradientTape() as tape:
            gen_cont = data.cont
            print(gen_cont.shape)
            gen_cont_feat = content_extract(model,gen_cont,content_layers)
            loss = content_loss(gen_cont_feat, true_cont_feat)

        grads = tape.gradient(loss, data.trainable_variables)
        optimizer.apply_gradients(zip(grads, data.trainable_variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    print("hi")


    #pass through true image and save output of conv layers
    #might need to change this to a different block to prevent loss of structural detail

    gen_img = np.squeeze(data.cont)

    plt.imshow(gen_img, interpolation= 'nearest')
    plt.show()
    plt.savefig('rand1.png')




if __name__ == "__main__":
    main()
