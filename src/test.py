from itertools import izip
import argparse
import os 
from keras.models import load_model
import skimage.io as io 
import pdb 
import numpy as np 
from sklearn.metrics import precision_recall_fscore_support

def CheckAndCreate(path): 
    if not os.path.exists(path): 
        os.makedirs(path)

def GenerateStats(xpath, ypath, model, outDir, threshold):
    xtest, ytest = io.imread(xpath), io.imread(ypath) 
    xtest, ytest = xtest[np.newaxis,:,:,np.newaxis], (ytest[np.newaxis,:,:,np.newaxis]).astype(bool)
    ypred = model.predict(xtest, verbose=0)
    
    
    io.imsave(os.path.join(outDir,'0.png'), ypred[0,:,:,0])

    p,r,f,s = precision_recall_fscore_support(ytest.flatten(), ypred.flatten() > opts.decideThreshold,
                                             beta=1.0, labels=[False,True])
    pdb.set_trace()
    return 

def Test(opts): 
    #model loading..
    model = load_model(opts.modelPath, compile=False)
    
    #pathnames loading for test images and labels
    directory = os.path.join(opts.dataDir, opts.dataType)
    fnamesX = sorted(os.listdir(os.path.join(directory, 'X')))
    pathnamesX = [os.path.join(directory,'X',f) for f in fnamesX if f.split('.')[-1] in opts.ext]

    fnamesY = sorted(os.listdir(os.path.join(directory,'Y')))
    pathnamesY = [os.path.join(directory,'Y',f) for f in fnamesY if f.split('.')[-1] in opts.ext]

    CheckAndCreate(opts.outDir)

    #testing image one by one..
    for xpath,ypath in izip(pathnamesX, pathnamesY): 
        GenerateStats(xpath, ypath, model, opts.outDir, opts.decideThreshold)
        break 
    return 

def SetArguments(parser): 
    #Data loading/saving parameters
    parser.add_argument('-dataDir',action='store', type=str, default='../data/generated/test', dest='dataDir')
    parser.add_argument('-dataType',action='store', type=str, default='noNoise', dest='dataType')
    parser.add_argument('-ext', action='store',type=list, default=['png', 'jpg'], dest='ext')
    parser.add_argument('-outRootDir', action='store',type=str, default='../results/', dest='outRootDir')

    #Model parameters 
    parser.add_argument('-expRootDir',action='store',type=str, default='../experiments/',dest='expRootDir')
    parser.add_argument('-modelExpName', action='store',type=str, default='strftime("%d-%m-%Y__%H-%M-%S",gmtime())', dest='modelExpName')

    #Other parameters 
    parser.add_argument('-decideThreshold', action='store',type=float, default=.5, dest='decideThreshold')
    parser.add_argument('-batchSize', action='store',type=int, default=1, dest='batchSize')
    parser.add_argument('-verbosity', action='store',type=int, default=1, dest='verbosity')
    return 

def PostprocessOpts(opts): 
    opts.outDir = os.path.join(opts.outRootDir, opts.modelExpName)
    opts.modelPath = os.path.join(opts.expRootDir, opts.modelExpName)
    opts.modelPath = os.path.join(opts.modelPath, 'model', os.listdir(os.path.join(opts.modelPath, 'model'))[0])
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    SetArguments(parser)

    opts = parser.parse_args()
    PostprocessOpts(opts)

    Test(opts)