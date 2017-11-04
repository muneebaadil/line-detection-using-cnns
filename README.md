# Hough Transform using Convolutional Neural Networks (CNNs)

Given a colored/gray-scale image, the task is to detect and localize basic geometric shapes (straight lines, circles, ellipses etc) using convolutional neural networks.

Standard HT (Hough Transform) populates an accumulator array whose size determines the level of detail we want to have. This introduces a tradeoff between precision and computational cost. Furthermore, populating an accumulator array is often too costly for realtime applications. Also, standard HT is not robust to noise i.e discontinuity of lines in pixel-space caused by discretization often votes the false parameters of the lines in the accumulator space.

This project aims to eliminate above mentioned limitations of classical Hough Transform. 


## 1. Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### 1.1. Prerequisites

You need to have following libraries installed:
```
Skimage >= 0.13.0
Numpy >= 1.13.1
Keras 
Tensorflow
```

## 2. Training

### 2.1. Dataset Preparation

1. Place your original images in ```./data``` folder

2. run the ```./code/prepare_dataset.py``` script:
```
cd src 
python prepare_dataset.py
``` 

This will do the following: 
1. Train/validation split 
2. Apply classical Hough Transform to generate ground-truth labels of each image

(See the function ```set_arguments()``` in the file ```prepare_dataset.py``` for information on arguments)

### 2.2. Training the model

## Authors

**Muneeb Aadil** - [Github Profile](https://github.com/muneebaadil) - (Email: imuneebaadil@gmail.com)

See also the list of [contributors](https://github.com/muneebaadil/Hough-Transform-using-CNNs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details