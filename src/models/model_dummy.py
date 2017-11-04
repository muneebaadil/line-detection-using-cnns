import keras.layers as layers 
from keras.initializers import VarianceScaling

def create_net(opts):
    """Simply two layer convolutional neural network for initial testing purposes"""
    
    init_func = VarianceScaling(2.)
    inputs = layers.Input(shape=(256,256,n_channels))
    
    c1 = layers.Conv2D(32,kernel_size=3,strides=1,padding='valid',activation='relu',
                            kernel_initializer=init_fn)(inputs)
    c2 = layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', activation='relu',
                        kernel_initializer=init_fn)(c1)

    model = Model(inputs=inputs,outputs=c2)
return model