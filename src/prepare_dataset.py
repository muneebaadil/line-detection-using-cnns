import skimage as skimg
from skimage.feature import canny
from skimage.color import rgb2ycbcr
import skimage.io as io 
import numpy as np 
import os 
from shutil import copyfile
import argparse 

def check_and_create(path): 
    if not os.path.exists(path): 
        os.makedirs(path)


def prepare_labels(pathnames, outfolderpath, printevery, canny_sigma):
    imgs = io.ImageCollection(pathnames)

    i = 0 
    for i, (img, pathname) in enumerate(zip(imgs, imgs.files)): 
        #converting to different format for edge map computation 
        img_ycbcr = rgb2ycbcr(img)

        #generating canny edge map and saving the output to label's folder
        out = canny(img_ycbcr[:,:,0], sigma=canny_sigma)

        #filenaming settings
        fname = pathname.split('/')[-1]
        Ypath = os.path.join(outfolderpath, 'Y', fname)
        Xpath = os.path.join(outfolderpath, 'X', fname)

        #moving the original picture to input's folder
        copyfile(pathname, Xpath)
        io.imsave(Ypath, out) 

        if (printevery>0) and (i%printevery==0): 
            print '{} files processed'.format(i)

def prepare_dataset(data_dir, ext, train_fraction, printevery, canny_sigma): 
    #creating training dataset folders..
    check_and_create(os.path.join(data_dir, 'train', 'X'))
    check_and_create(os.path.join(data_dir, 'train', 'Y'))

    #validation dataset folders..
    check_and_create(os.path.join(data_dir, 'val', 'X'))
    check_and_create(os.path.join(data_dir, 'val', 'Y'))
    
    #reading image filenames..
    fnames = os.listdir(data_dir)
    pathnames = [data_dir+f for f in fnames if f.split('.')[-1] in ext]
    
    #shuffling and splitting training/test set..
    np.random.shuffle(pathnames)
    train_pathnames = pathnames[:int(train_fraction*len(pathnames))]
    val_pathnames = pathnames[int(train_fraction*len(pathnames)):]
    
    #preparing training set
    prepare_labels(train_pathnames, os.path.join(data_dir, 'train'), printevery, canny_sigma)
    
    #preparing validation set
    prepare_labels(val_pathnames, os.path.join(data_dir, 'val'), printevery, canny_sigma)
    
    return 


def set_arguments(parser):
    parser.add_argument('-dataDir',action='store', type=str, default='../data/', dest='dataDir')
    parser.add_argument('-trainFraction', action='store', type=float, default=.8, dest='trainFraction')
    parser.add_argument('-cannySigma', action='store', type=float, default=1., dest='cannySigma')
    parser.add_argument('-ext', action='store',type=list, default=['png', 'jpg'], dest='ext')
    parser.add_argument('-printEvery', action='store', type=int, default=-1, dest='printEvery')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    set_arguments(parser)
    opts = parser.parse_args()

    prepare_dataset(opts.dataDir, opts.ext, opts.trainFraction, opts.printEvery, opts.cannySigma)