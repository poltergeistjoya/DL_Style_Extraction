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
flags.DEFINE_integer("num_iters", 50000, "number of iterations for ADAM")
flags.DEFINE_string("optype", "style", "style or content extraction")
flags.DEFINE_string("impath", "starry_night.jpeg", "path to content or style image")


#make trainavar for gen content and style
class Data(tf.Module):
    def __init__(self, rng):

        self.cont = tf.Variable(rng.uniform(low=0.0,high=1.0,size= [1,224,224,3]), trainable = True, dtype=tf.float32)


#crop,resize, normalize image to 224 by 224, vals 0 to 1
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


#content loss, gen_cont, true_cont
def content_loss(gen_cont, true_cont):
    return 0.5*tf.math.reduce_sum(tf.math.square(true_cont - gen_cont))

#for style loss
def gram_matrix(input_tensor): # from https://www.tensorflow.org/tutorials/generative/style_transfer
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  #print(result)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


#style loss function, generated content, true content
def style_loss(gen_cont, true_cont):
  per_layer_loss = []
  loss = 0
  if len(gen_cont) != len(true_cont):
    sys.exit('Style_loss requires same number of layers')
  for layer_num in range(len(gen_cont)):
    h,w = gen_cont[layer_num].shape[1], gen_cont[layer_num].shape[2]
    if layer_num <= 1:  # first 2 layers have 2 sublayers
      num_sub_layer = 2
    else:
      num_sub_layer = 3
    loss += tf.math.reduce_sum(((gram_matrix(gen_cont[layer_num]) - gram_matrix(true_cont[layer_num])) /(2 * h * w * num_sub_layer  ))**2)

  return loss

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

def style_extract(model, data, style_layers):
  outputs = [model.get_layer(name).output for name in style_layers] # loop through style_layers
  feature_extractors= tf.keras.Model([model.input], outputs)
  features = feature_extractors(data)
  return features

def main():

    #parse flags before use
    FLAGS(sys.argv)
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    lr = FLAGS.lr
    iters = FLAGS.num_iters
    optype = FLAGS.optype
    path = FLAGS.impath

    #random number gen
    np_rng = np.random.default_rng(31415)

    #true image to get content and style
    true_img = img_resize(path)
    true = img_to_VGG(true_img)

    #randomly generated noise
    data = Data(np_rng)

    #initialize vgg model
    model = vgg_16()


    boundaries = [300, 800, 2700, 3200, 5500, 12000]
    values=[.1, .05, .025, .01, .005, .003,.001]

    lr_schedule=keras.optimizers.schedules.PiecewiseConstantDecay(boundaries,values)

    optimizer = keras.optimizers.Adam(learning_rate = lr_schedule)

    #content layers
    content_layers = 'block3_conv1'

    #style layers
    style_layers =['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

    #get content features
    true_cont_feat=content_extract(model, true, content_layers)

    #get style features
    true_style_feat = style_extract(model, true, style_layers)

    gen_cont = np.squeeze(data.cont)
    plt.imshow(gen_cont, interpolation='nearest')
    plt.show()
    plt.savefig('rand.png')

    if optype =="content":

        lossarr = np.zeros(iters, dtype=float)
        bar = trange(iters)

        for i in bar:
            with tf.GradientTape() as tape:
                gen_cont = data.cont
                sig_gen_cont = tf.math.sigmoid(gen_cont)
                gen_cont_feat = content_extract(model,gen_cont,content_layers)
                loss = content_loss(gen_cont_feat, true_cont_feat)

                grads = tape.gradient(loss, data.trainable_variables)
                optimizer.apply_gradients(zip(grads, data.trainable_variables))

            lossarr[i] = loss.numpy()
            bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
            bar.refresh()

        gen_img = np.squeeze(data.cont)
        sig_gen_img = tf.math.sigmoid(gen_img)

        plt.imshow(sig_gen_img, interpolation= 'nearest')
        plt.show()
        plt.savefig('content.png')


        lossarr = np.expand_dims(lossarr, axis = 1)
        plt.figure()
        ax = plt.gca()

        ax.set_ylim([-1,5000])
        plt.plot(np.arange(iters), lossarr, color ="red")
        plt.title("Loss at iteration")
        plt.tight_layout()
        plt.savefig("./lossiter")


    #style transfer
    #yea, i know its WET
    if optype == "style":
        iters = 1000;
        boundaries = [300]
        values=[.05, .01]

        lr_schedule=keras.optimizers.schedules.PiecewiseConstantDecay(boundaries,values)
        optimizer = keras.optimizers.Adam(learning_rate = lr_schedule)

        lossarr = np.zeros(iters, dtype=float)
        bar = trange(iters)
        for i in bar:
            with tf.GradientTape() as tape:
                gen_cont = data.cont
                sig_gen_cont = tf.math.sigmoid(gen_cont)
                gen_style_feat = style_extract(model, sig_gen_cont, style_layers)
                loss = style_loss(gen_style_feat, true_style_feat)

            grads = tape.gradient(loss, data.trainable_variables)
            optimizer.apply_gradients(zip(grads, data.trainable_variables))

            bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
            bar.refresh()

        lossarr[i] = loss.numpy()
        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        gen_img = np.squeeze(data.cont)
        sig_gen_img = tf.math.sigmoid(gen_img)

        plt.imshow(sig_gen_img, interpolation= 'nearest')
        plt.show()
        plt.savefig('style.png')

        lossarr = np.expand_dims(lossarr, axis = 1)
        plt.figure()
        ax = plt.gca()

        ax.set_ylim([0,.01])
        plt.plot(np.arange(iters), lossarr, color ="red")
        plt.title("Loss at iteration")
        plt.tight_layout()
        plt.savefig("./lossiter")



if __name__ == "__main__":
    main()
