# Hough Transform using Convolutional Neural Networks (CNNs)

Given a binary edge-image, the task is to detect and localize basic geometric shapes (straight lines, circles, ellipses etc) using convolutional neural networks.

Standard HT (Hough Transform) populates an accumulator array whose size determines the level of detail we want to have. This introduces a tradeoff between precision and computational cost. Furthermore, populating an accumulator array is often too costly for realtime applications. Also, standard HT is not robust to noise i.e discontinuity of lines in pixel-space caused by discretization often votes the false parameters of the lines in the accumulator space.

This project aims to eliminate above mentioned limitations of classical Hough Transform. 


## 1. Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### 1.1. Prerequisites

You need to have following libraries installed:
```
Skimage >= 0.13.0
Sklearn >= 0.19.1
Numpy >= 1.13.1

Tensorflow >= 1.0.0
Keras >= 2.0.5 
Keras_contrib >= 0.0.2

Pydot
Graphviz
```

### 1.2. Installation

#### 1.2.1. Anaconda

Although, packages listed above can be seperately downloaded and installed, it's recommended to install Anaconda package to install all scipy libraries at once.

1. Download Anaconda Installer from [here](https://www.anaconda.com/download/)

2. Run the downloaded ```.sh``` script with bash: 
```bash Anaconda****.sh```

#### 1.2.2. Keras 
Use ```conda``` package manager to install Keras:
  * For CPU version: ```conda install keras```
  * For GPU Version: ```conda install -c anaconda keras-gpu``` 
(Note: This will automatically install ```tensorflow``` too.)

## 2. Training

### 2.1. Dataset Preparation

1. Go to ```./src/``` folder
```cd src/```

2. Run the ```generate_dataset.py``` script to generate the dataset synthetically:
```python prepare_dataset.py -outDir <path-to-save-images> -numImgs <number-of-images-to-generate>``` 

(See all arguments using ```python generate_dataset.py --help```)

This will generate and save input images and ground truth respective images at ```outDir/X/``` and ```outDir/Y/```

### 2.2. Training the model

Run the ```train.py``` script to train a model on generated dataset, like so: 
```python train.py -dataDir <path-to-dataset> -netType <network-name> -logDir <path-to-save-experiment>``` 
(See all arguments using ```python train.py --help```)

This will train a specified model on the specified dataset and will save the following to ```logRootDir/logDir/```: 
  * Model architecture along with weights
  * Tensorboard logs
  * Predictions on validation set (of best performing model only)
  * Options used for generating this experiment

## 3. Testing

## Authors

**Muneeb Aadil** - [Github Profile](https://github.com/muneebaadil) - (Email: imuneebaadil@gmail.com)

See also the list of [contributors](https://github.com/muneebaadil/Hough-Transform-using-CNNs/contributors) who participated in this project.