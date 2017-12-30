from itertools import izip
import argparse
import os 
from keras.models import load_model
import skimage.io as io 
import pdb 
import numpy as np 
from sklearn.metrics import precision_recall_fscore_support
from utils import * 
from skimage import img_as_float
#import matplotlib.pyplot as plt

def Test(opts): 
    #model loading..
    model = load_model(opts.modelPath, compile=False)
    
    #pathnames loading for test images and labels
    directory = os.path.join(opts.dataDir, opts.dataType)
    fnamesX = sorted(os.listdir(os.path.join(directory, 'X')))
    pathnamesX = [os.path.join(directory,'X',f) for f in fnamesX if f.split('.')[-1] in opts.ext]

    fnamesY = sorted(os.listdir(os.path.join(directory,'Y')))
    pathnamesY = [os.path.join(directory,'Y',f) for f in fnamesY if f.split('.')[-1] in opts.ext]

    #loading images and model 
    imgsX = io.ImageCollection(load_pattern=pathnamesX)
    imgsY = io.ImageCollection(load_pattern=pathnamesY)

    imgsX = img_as_float(imgsX.concatenate()[:,:,:,np.newaxis])
    imgsY = img_as_float(imgsY.concatenate()[:,:,:,np.newaxis])

    model = load_model(opts.modelPath,compile=False)

    predY = model.predict(imgsX, batch_size=opts.batchSize, verbose=opts.verbosity)

    p,r,f,_ = precision_recall_fscore_support(y_true=imgsY.flatten(), y_pred=predY.flatten() > opts.decideThreshold,
                                    labels=[False,True])

    print '[decideThreshold = {}] precision={}, recall={}, fscore(beta=1.)={}'.format(opts.decideThreshold,p,r,f,_)
    print 'precision_avg={}, recall_avg={}, fcore_avg(beta=1.)={}'.format(p.mean(),r.mean(),f.mean())

    if opts.saveTestOutput==True: 
        CheckAndCreate(opts.outDir)

        for predY_,fnameX in izip(predY,fnamesX): 
            io.imsave(os.path.join(opts.outDir,fnameX), predY_[:,:,0])

    if opts.saveAnalysis==True: 
        CheckAndCreate(opts.outDir)

        for ypred, yout in izip(predY,imgsY): 
            analysis = np.zeros((yout.shape[0],yout.shape[1],3))

            ypred=ypred[:,:,0]
            yout=yout[:,:,0]

            p,r,f,_=precision_recall_fscore_support(yout.flatten(),ypred.flatten()>opts.decideThreshold,
                                                    labels=[False,True])
            
            analysis[(ypred>.5)==False] = np.array([0.,0.,0.])
            analysis[(ypred>.5)==True] = np.array([1.,1.,1.])
            analysis[np.logical_and((ypred>.5)==True, yout==False)] = np.array([1.,0.,0.])
            analysis[np.logical_and((ypred>.5)==False, yout==True)] = np.array([0.,0.,1.])

            io.imsave(os.path.join(opts.outDir,'analysis_'+fnameX), analysis)
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
    parser.add_argument('-saveTestOutput', action='store',type=bool, default=False, dest='saveTestOutput')
    parser.add_argument('-saveAnalysis', action='store',type=bool, default=False, dest='saveAnalysis')
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