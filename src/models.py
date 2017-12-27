#THIS FILE CONTAINS ALL THE MODEL ARCHITECTURES EXPERIMENTED WITH.. 

import keras.layers as layers 
from keras.initializers import VarianceScaling
from keras.models import Model, Sequential
import itertools as iters

def model_init(opts):
    """Simple sequential image-to-image convolutional neural network"""
    
    init_fn = VarianceScaling(2.)

    model = Sequential()
    isFirst = True
    for ks, nk, a in iters.izip(opts.kernelSizes, opts.numKernels, opts.activations): 

        if isFirst:
            model.add(layers.Conv2D(nk,kernel_size=ks,strides=opts.strides,padding=opts.padding,activation=a,
                                kernel_initializer=init_fn, input_shape=opts.inputShape))
            isFirst = False
        else: 
            model.add(layers.Conv2D(nk,kernel_size=ks,strides=opts.strides,padding=opts.padding,activation=a,
                                kernel_initializer=init_fn))

        if opts.dropRate > 0.0: 
            model.add(layers.Dropout(rate=opts.dropRate))
    return model

models_dict = dict()
models_dict['imageToImageSeq'] = model_init