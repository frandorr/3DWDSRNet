import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, Model, regularizers
import tensorflow_addons as tfa

MEAN = 7433.6436
STD = 2353.0723
LR_SIZE = 34


def normalize(x):
    return (x-MEAN)/STD

def denormalize(x):
    return x * STD + MEAN 


def conv3d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(layers.Conv3D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(layers.Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

def reflective_padding(name):
    return layers.Lambda(lambda x: tf.pad(x,[[0,0],[1,1],[1,1],[0,0],[0,0]],mode='REFLECT',name=name))

def res_block_b(x_in, num_filters, expansion, kernel_size):
    linear = 0.8
    x = conv3d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    x = conv3d_weightnorm(int(num_filters * linear), 1, padding='same')(x)
    x = conv3d_weightnorm(num_filters, kernel_size, padding='same')(x)
    return x_in + x

def wdsr_3d(scale, num_filters, num_res_blocks, res_block_expansion, res_block,channels):
    img_inputs = Input(shape=(LR_SIZE, LR_SIZE,channels, 1))
    mean_inputs = Input(shape=(LR_SIZE,LR_SIZE,1))

    x = layers.Lambda(normalize)(img_inputs)
    y = layers.Lambda(normalize)(mean_inputs)
    
    # Main path
    m = conv3d_weightnorm(num_filters, (3,3,3),padding='same',activation='relu')(x)
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3)
        
    for i in range(0,np.floor_divide(channels,3)):
        m = reflective_padding(name="ref_padding_{}".format(i))(m)
        m = conv3d_weightnorm(num_filters, (3,3,3), padding='valid',activation='relu',
                              name="Conv_Reducer_{}".format(i))(m)
    # Upscaling main path   
    m = conv3d_weightnorm(scale ** 2, (3,3,3),
                      padding='valid')(m)
    
    m = layers.Reshape((LR_SIZE-2,LR_SIZE-2,9))(m)
    m = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 3))(m)
    
    # Residual path
    y = layers.Lambda(lambda x: tf.pad(x,[[0,0],[1,1],[1,1],[0,0]],mode='REFLECT',name='padding_2d'))(y)
    # Upscaling residuale path
    y = conv2d_weightnorm(scale ** 2, (3,3),padding='valid', activation='relu')(y)
    y = conv2d_weightnorm(scale ** 2, (3,3),padding='valid')(y)

    y = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 3))(y)
    outputs = y+m
    outputs = layers.Lambda(denormalize)(outputs)

    return Model([img_inputs,mean_inputs], outputs, name="wdsr")