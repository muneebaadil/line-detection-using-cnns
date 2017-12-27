import argparse
import os 
from keras.models import load_model
import skimage.io as io 

def LoadData(directory, ext): 

    #Determining pathnames of X and Y images
    fnamesX = sorted(os.listdir(os.path.join(directory, 'X')))
    pathnamesX = [os.path.join(directory,'X',f) for f in fnamesX if f.split('.')[-1] in ext]

    fnamesY = sorted(os.listdir(os.path.join(directory,'Y')))
    pathnamesY = [os.path.join(directory,'Y',f) for f in fnamesY if f.split('.')[-1] in ext]

    #Loading Images
    X = io.ImageCollection(pathnamesX).concatenate()
    Y = io.ImageCollection(pathnamesY).concatenate()

    return X,Y

def test(opts): 
    model = load_model(opts.modelPath)

    X,Y = LoadData(os.path.join(opts.dataDir, opts.dataType), opts.ext)
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
    parser.add_argument('-batchSize', action='store',type=float, default=.5, dest='batchSize')
    return 

def PostprocessOpts(opts): 
    opts.outDir = os.path.join(opts.outRootDir, opts.modelExpName)
    opts.modelPath = os.path.join(os.expRootDir, opts.modelExpName, os.listdir(opts.modelExpName, 'model')[0])
    return

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
	SetArguments(parser)

	opts = parser.parse_args()
	PostprocessOpts(opts)

	test(opts)