import skimage as skimg
from skimage.feature import canny
from skimage.color import rgb2ycbcr
import skimage.io as io 
from skimage.transform import hough_line, hough_line_peaks
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
        img_hsv = rgb2hsv(img)

        #generating canny edge map
        Ximage = canny(img_hsv[:,:,-1], sigma=canny_sigma)
        Yimage = generate_hough_label(Ximage)

        #filenaming settings
        fname = pathname.split('/')[-1]
        Ypath = os.path.join(outfolderpath, 'Y', fname)
        Xpath = os.path.join(outfolderpath, 'X', fname)

        #saving input and label to disk
        io.imsave(Xpath, Ximage)
        io.imsave(Ypath, Yimage)
        
        #optional verbosity
        if (printevery>0) and (i%printevery==0): 
            print '{} files processed'.format(i)

def prepare_dataset(in_data_dir, out_data_dir, ext, train_fraction, printevery, canny_sigma, takeonly=None): 
    """
    
    """
    
    #creating training dataset folders..
    check_and_create(os.path.join(out_data_dir, 'train', 'X'))
    check_and_create(os.path.join(out_data_dir, 'train', 'Y'))

    #validation dataset folders..
    check_and_create(os.path.join(out_data_dir, 'val', 'X'))
    check_and_create(os.path.join(out_data_dir, 'val', 'Y'))
    
    #reading image filenames..
    fnames = os.listdir(in_data_dir)
    if takeonly: 
        fnames = fnames[:takeonly]
    pathnames = [in_data_dir+f for f in fnames if f.split('.')[-1] in ext]
    
    #shuffling and splitting training/test set..
    np.random.shuffle(pathnames)
    cutoff = int(train_fraction*len(pathnames))
    train_pathnames = pathnames[:cutoff]
    val_pathnames = pathnames[cutoff:]
    
    #preparing training set
    prepare_labels(train_pathnames, os.path.join(out_data_dir, 'train'), printevery, canny_sigma)
    
    #preparing validation set
    prepare_labels(val_pathnames, os.path.join(out_data_dir, 'val'), printevery, canny_sigma)
    
    return 

def generate_hough_label(img): 
    """
    Given a binary edge-image, generates a hough output i.e turns off the edge pixels that 
    do not fall on any line, and keeps the edge pixels on otherwise. 
    
    Args: 
    img: binary edge image

    Returns: 
    out: line-edges output image
    """
    out, out_temp = np.zeros_like(img, dtype=np.float32), np.zeros_like(img, dtype=np.float32)
    
    #calculating hough-space, and converting to probabilities
    h, theta, d = hough_line(img)
    h = h/float(np.max(h)) 
    
    for val, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        #calculating end coordinates of line
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
        
        #generating x and corresponding y coordinates
        xidx = np.arange(0, img.shape[1])
        yidx = np.linspace(y0, y1, num=xidx.shape[0]).astype(np.int)
        
        #filtering out point not in image space
        yidx_valid_idx = np.nonzero((yidx >= 0) & (yidx < img.shape[0]))
        xidx = xidx[yidx_valid_idx]
        yidx = yidx[yidx_valid_idx]
        
        out_temp[yidx, xidx] = val
        
    #removing the lines corresponding to edges not present in input edge-map 
    keep_idx = (out_temp.astype(np.bool) & img)
    out_temp[np.logical_not(keep_idx)] = 0 
    
    return out_temp 

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