#THIS FILE CONTAINS ALL THE MODEL ARCHITECTURES EXPERIMENTED WITH.. 

import keras.layers as layers 
from keras.initializers import VarianceScaling
from keras.models import Model, Sequential

def model_init(opts):
    """Simply two layer convolutional neural network for initial testing purposes"""
    
    init_fn = VarianceScaling(2.)
    # inputs = layers.Input(shape=(256,256,3))
    
    # c1 = layers.Conv2D(32,kernel_size=3,strides=1,padding='valid',activation='relu',
    #                         kernel_initializer=init_fn)(inputs)
    # if opts.dropout >= 0: 
        
    # c2 = layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', activation='relu',
    #                     kernel_initializer=init_fn)(c1)

    # model = Model(inputs=inputs,outputs=c2)
    # return model

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
    

models_dict = dict()
models_dict['model_init'] = model_init