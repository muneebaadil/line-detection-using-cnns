#THIS FILE CONTAINS ALL THE MODEL ARCHITECTURES EXPERIMENTED WITH.. 

import keras.layers as layers 
from keras.initializers import VarianceScaling
from keras.models import Model, Sequential

def model_init(opts):
    """Simply two layer convolutional neural network for initial testing purposes"""
    
    init_fn = VarianceScaling(2.)

    model = Sequential()

    model.add(layers.Conv2D(32,kernel_size=3,strides=1,padding='same',activation='relu',
                             kernel_initializer=init_fn, input_shape=(512,512,1)))
    if opts.dropRate >= 0: 
        model.add(layers.Dropout(rate=opts.dropRate))
    
    model.add(layers.Conv2D(32,kernel_size=3,strides=1,padding='same',activation='relu',
                             kernel_initializer=init_fn))
    if opts.dropRate >= 0: 
        model.add(layers.Dropout(opts.dropRate))

    model.add(layers.Conv2D(1,kernel_size=1,strides=1,padding='same',activation='relu',
                            kernel_initializer=init_fn))
    return model    

models_dict = dict()
models_dict['model_init'] = model_init