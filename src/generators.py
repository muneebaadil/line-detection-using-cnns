#THIS FILE CONTAINS THE DATA_LOADERS/GENERATORS AND PREPROCESSING TECHNIQUES WE EXPERIMENTED WITH
import skimage.io as io
import os 
import numpy as np 


def generator_full_image(directory, ext, batch_size, preprocessing=None, mode='train'):
    """"""
    #loading filenames (only and not the image)
    fnamesX = os.listdir(os.path.join(directory, 'X'))
    pathnamesX = [os.path.join(directory,'X',f) for f in fnamesY if f.split('.')[-1] in ext]

    fnamesY = os.listdir(os.path.join(directory,'Y'))
    pathnamesY = [os.path.join(directory,'Y',f) for f in fnamesY if f.split('.')[-1] in ext]
    
    idx = np.arange(len(pathnames))
    while True:

        #shuffling at the start of each epoch..
        np.random.shuffle(idx)

        for i in xrange(0, len(pathnames), batch_size): 
            
            #picking a batch.. 
            pathnamesX_batch = pathnamesX[idx[i:i+batch_size]]
            pathnamesY_batch = pathnamesY[idx[i:i+batch_size]]

            imagesX = io.ImageCollection(pathnamesX_batch)
            imagesY = io.ImageCollection(pathnamesY_batch)
            
            #optionally applying preporcessing to each function
            if preprocessing is not None: 
                imagesX, imagesY = preprocessing(imagesX, imagesY)

            yield imagesX, imagesY

generators_dict = dict()
generators_dict['generator_full_image'] = generator_full_image