#THIS FILE CONTAINS ALL THE LOSS FUNCTIONS WE EXPERIMENTED WITH.. 

import keras.metrics as mts 

losses_dict = dict()
losses_dict['binary_crossentropy'] = mts.binary_crossentropy