#THIS FILE CONTAINS ALL THE MODEL ARCHITECTURES EXPERIMENTED WITH.. 

import keras.layers as layers
from keras.initializers import VarianceScaling
from keras.models import Model, Sequential
import itertools as iters
from keras_contrib.layers.normalization import InstanceNormalization

def model_init(opts):
    """Simple sequential image-to-image convolutional neural network"""
    
    init_fn = VarianceScaling(2.)

    model = Sequential()
    isFirst = True
    for ks, nk, a in iters.izip(opts.kernelSizes, opts.numKernels, opts.activations): 

        if isFirst:
            model.add(layers.Conv2D(nk,kernel_size=ks,strides=opts.strides,padding=opts.padding,
                                kernel_initializer=init_fn, input_shape=opts.inputShape))
            isFirst = False
        else: 
            model.add(layers.Conv2D(nk,kernel_size=ks,strides=opts.strides,padding=opts.padding,
                                kernel_initializer=init_fn))

        if opts.includeInsNormLayer: 
            model.add(InstanceNormalization(axis=opts.insNormAxis))

        model.add(layers.Activation(a))
        if opts.dropRate > 0.0: 
            model.add(layers.Dropout(rate=opts.dropRate))
    return model

def model_unet(opts): 
    init_fn = VarianceScaling(2.)
    model_input = layers.Input(shape=opts.inputShape)
    prev = model_input
    scaleSpace = []

    #downsampling path
    for i in xrange(opts.numScales): 
        out = layers.Conv2D(filters=opts.numKernels, kernel_size=opts.kernelSizes, strides=opts.strides, padding='same',
                    kernel_initializer=init_fn,activation=opts.activations)(prev)
        scaleSpace.append(out)
        
        prev = layers.MaxPool2D(pool_size=opts.poolSize, strides=opts.poolStrides, padding=opts.poolPadding)(out)

    #base case
    prev = layers.Conv2D(filters=opts.numKernels, kernel_size=opts.kernelSizes, strides=opts.strides, padding='same',
                        kernel_initializer=init_fn,activation=opts.activations)(prev)
    
    #upsampling path
    for i in xrange(opts.numScales): 
        out = layers.Conv2DTranspose(filters=opts.numKernels,kernel_size=opts.kernelSizes, strides=2, padding='same',
                                    kernel_initializer=init_fn,)(prev)
 
        out = layers.concatenate(inputs=[out, scaleSpace[-1-i]], axis=-1)
        prev = layers.Conv2D(filters=opts.numKernels, kernel_size=opts.kernelSizes, strides=opts.strides, padding='same',
                    kernel_initializer=init_fn,activation=opts.activations)(out)
    
    model_output = layers.Conv2D(filters=1, kernel_size=opts.kernelSizes, strides=opts.strides, padding='same',
                    kernel_initializer=init_fn,activation='sigmoid')(prev)

    model = Model(inputs=(model_input,), outputs=(model_output,))
    return model

models_dict = dict()
models_dict['imageToImageSeq'] = model_init
models_dict['imageToImageUnet'] = model_unet