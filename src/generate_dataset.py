import skimage.draw as draw 
import skimage.io as io 
import os 
import numpy as np 
import argparse 

import pdb

def check_and_create(path): 
    if not os.path.exists(path): 
        os.makedirs(path)
        
def AddLines(X,Y,maxNumLines,imgSize): 
    numLines = np.random.randint(1,maxNumLines)
    
    for i in xrange(numLines): 
        [a,b,c,d] = np.random.randint(0,imgSize[1], size=4)
        idx = draw.line(a,b,c,d)
        
        X[idx] = 1. 
        Y[idx] = 1.
    return 

def AddOthers(X,maxNumOthers,imgSize): 
    numOthers = np.random.randint(1,maxNumOthers)
    
    for i in xrange(numOthers):
        [r,c,radius] = np.random.randint(0,imgSize[1]/2,size=3)
        r = int(r*2) 
        c = int(c*2)
        idx = draw.circle_perimeter(r,c,radius,shape=imgSize)
        X[idx] = 1.
    return 

def GenerateExample(imgSize, maxNumLines, maxNumOthers):
    X,Y = np.zeros(imgSize), np.zeros(imgSize)
    
    #adding lines
    AddLines(X,Y,maxNumLines,imgSize)
    
    #adding other geometry objects i.e things NOT supposed to be
    #detected by hough transfrom 
    AddOthers(X,maxNumOthers,imgSize)
    
    return X,Y

def GenerateDataset(outDir, numImgs, imgSize, maxNumLines, maxNumOthers, printEvery=-1): 
    check_and_create(os.path.join(outDir, 'X'))
    Xpath = os.path.join(outDir, 'X')
    check_and_create(os.path.join(outDir, 'Y'))
    Ypath = os.path.join(outDir, 'Y')
    
    for i in xrange(numImgs): 
        #generating an example 
        x,y = GenerateExample(imgSize, maxNumLines, maxNumOthers)
        
        #saving to disk..
        pdb.set_trace()
        io.imsave(os.path.join(Xpath,'{}.png'.format(i)), x)
        io.imsave(os.path.join(Ypath,'{}.png'.format(i)), y)
    
        #optional verbosity
        if (printEvery > 0) and ((i % printEvery) == 0): 
            print '{}/{} images generated..'.format(outDir, numImgs)

def writeOptsToFile(fpath, optsDict): 
    fobj = open(fpath, 'w')

    for k,v in optsDict.items(): 
        fobj.write('{} >> {}\n'.format(str(k), str(v)))
    
    fobj.close()

def set_arguments(parser):
    parser.add_argument('-outDir',action='store', type=str, default='../data/generated2/train/noNoise', dest='outDir')
    parser.add_argument('-numImgs',action='store', type=int, default=10, dest='numImgs')
    parser.add_argument('-imgSize',action='store', type=int, default=512, dest='imgSize')
    parser.add_argument('-maxNumLines',action='store', type=int, default=10, dest='maxNumLines')
    parser.add_argument('-maxNumOthers',action='store', type=int, default=5, dest='maxNumOthers')
    parser.add_argument('-printEvery',action='store', type=int, default=1000, dest='printEvery')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    set_arguments(parser)
    opts = parser.parse_args()

    GenerateDataset(opts.outDir, opts.numImgs, (opts.imgSize,opts.imgSize), opts.maxNumLines, opts.maxNumOthers, opts.printEvery)
    writeOptsToFile(os.path.join(opts.outDir,'config.txt'), vars(opts))