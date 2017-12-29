#THIS FILE CONTAINS ALL THE MODEL ARCHITECTURES EXPERIMENTED WITH.. 

import keras.layers as layers
from keras.initializers import VarianceScaling
from keras.models import Model, Sequential
import itertools as iters
from keras_contrib.layers.normalization import InstanceNormalization

def ImageToImageSeqModels(opts):
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

# def UnetModels(opts): 
#     init_fn = VarianceScaling(2.)

#     scaleLayerInput = layers.Input(shape=opts.inputShape)
#     scaleSpace = []

#     #downsampling path
#     for i in xrange(opts.numScales): 
#         out = layers.Conv2D(opts.numKernels, kernel_size=opts.kernelSizes, strides=opts.strides, padding=opts.padding, 
#                     kernel_initializer=init_fn)(scaleLayerInput)

#         scaleSpace.append(out)
#         scaleLayerInput = layers.MaxPool2D(pool_size=opts.poolSize,strides=opts.strides,padding='valid')

#     #middle ground
#     scaleLayerInput = layers.Conv2D(opts.numKernels, kernel_size=opts.kernelSizes, strides=opts.strides, padding='same', 
#                                     kernel_initializer=init_fn)(scaleLayerInput)
    
#     #upsampling path
#     for i in xrange(opts.numScales): 
#         out = layers.Up

models_dict = dict()
models_dict['imageToImageSeq'] = ImageToImageSeqModels