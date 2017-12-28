import keras.callbacks as callbacks 
import numpy as np
import os
import skimage.io as io 
from utils import *
from skimage import img_as_float
import pdb

class valImagesSaver(callbacks.Callback): 
    def __init__(self, dataDir, ext, outDir): 
        callbacks.Callback.__init__(self)
        self.min = np.inf
        self.dataDir = dataDir
        self.outDir = outDir

        CheckAndCreate(outDir)

        fnamesX = sorted(os.listdir(self.dataDir))
        pathnamesX = [os.path.join(dataDir,f) for f in fnamesX if f.split('.')[-1] in ext]
        self.pathnamesX = np.array(pathnamesX)

    def on_batch_end(self, epoch, logs): 

        #If best validation model yet..
        if logs.get('val_loss') < self.min:
            self.min = logs.get('val_loss')
            
            # for path in self.pathnamesX: 
            #     img = io.imread(path)
            #     img = img[np.newaxis, :, :, np.newaxis]
                
            #     ypred = self.model.predict(img)

            #     fname = path.split('/')[-1]
            #     outPath = os.path.join(self.outDir, fname)

            #     io.imsave(outPath, ypred[0,:,:,0])

            imgs = io.ImageCollection(self.pathnamesX)
            imgs = img_as_float(imgs.concatenate()[:,:,:,np.newaxis])

            ypreds = self.model.predict(imgs,batch_size=1)

            for i, path in enumerate(self.pathnamesX): 
                fname = path.split('/')[-1]
                outPath = os.path.join(self.outDir, fname)

                pdb.set_trace()
                io.imsave(outPath, ypreds[i,:,:,0])