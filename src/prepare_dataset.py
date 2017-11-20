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

def my_hough_line(img, printevery):
    '''Perform a straight line Hough transform.

    Args:
    img: (M, N) ndarray input edge-map

    Returns
    accumulator
    thetas 
    rhos
    lines_cache_[x|y]coords.'''
    
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
    lines_cache_xcoords, lines_cache_ycoords = defaultdict(list), defaultdict(list)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[rho, t_idx] += 1
            
            lines_cache_xcoords[(rho, thetas[t_idx])].append(x)
            lines_cache_ycoords[(rho, thetas[t_idx])].append(y)
        
        if (printevery >= 1)and ((i%printevery)==0):
            print '{}/{} done..'.format(i, len(x_idxs))

    return accumulator, thetas, rhos, lines_cache_xcoords, lines_cache_ycoords

def generate_hough_label(img, printevery=-1): 
    """
    Given a binary edge-image, generates a hough output i.e turns off the edge pixels that 
    do not fall on any line, and keeps the edge pixels on otherwise. 
    
    Args: 
    img: binary edge image

    Returns: 
    edge_map: line-edges output image
    """
    edge_map = np.zeros_like(img_test)
    
    acc,thetas,rhos,lines_xcoords,lines_ycoords=my_hough_line(img,printevery)
    height,width = img.shape
    diag_len = np.ceil(np.sqrt(width * width + height * height))
    
    for val, angle, dist in zip(*hough_line_peaks(acc,thetas,rhos)):
        y_coords = lines_ycoords[(round(dist)+diag_len, angle)]
        x_coords = lines_xcoords[(round(dist)+diag_len, angle)]
        edge_map[y_coords,x_coords] = val/float(diag_len) #probabilites conversion        
    return edge_map

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