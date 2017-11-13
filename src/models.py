#THIS FILE CONTAINS ALL THE MODEL ARCHITECTURES EXPERIMENTED WITH.. 

import keras.layers as layers 
from keras.initializers import VarianceScaling
from keras.models import Model

def model_init(opts):
    """Simply two layer convolutional neural network for initial testing purposes"""
    
    init_fn = VarianceScaling(2.)
    inputs = layers.Input(shape=(256,256,3))
    
    c1 = layers.Conv2D(32,kernel_size=3,strides=1,padding='valid',activation='relu',
                            kernel_initializer=init_fn)(inputs)
    c2 = layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', activation='relu',
                        kernel_initializer=init_fn)(c1)

    model = Model(inputs=inputs,outputs=c2)
    return model

models_dict = dict()
models_dict['model_init'] = model_init