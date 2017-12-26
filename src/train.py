import argparse
import numpy as np
import importlib
import os 
from time import gmtime, strftime

from models import models_dict
from optimizers import optimizers_dict
from generators import generators_dict
from keras.callbacks import ModelCheckpoint

import pdb 

def writeConfigToFile(fpath, optsDict, model): 
    fobj = open(fpath, 'w')

    for k,v in optsDict.items(): 
        fobj.write('{} >> {}\n'.format(str(k), str(v)))
    fobj.write('\nmodel json:\n{}'.format(model.to_json()))
    fobj.close()

def train(opts): 
	"""Performs the whole algorithm i.e trains a given neural network on given data using given learning parameters
	
	Args: 
	opts: command line arguments
	
	Returns: 
	None"""

	#Creating given model..
	model = models_dict[opts.netType](opts)
	
	#Compiling given model using given learning parameters..
	optimizer = optimizers_dict[opts.optimizerType](lr=opts.learningRate, decay=opts.lrDecay)
	model.compile(optimizer=optimizer, loss=opts.lossType)
	
	#Configuring data loaders/generators now..
	train_generator = generators_dict[opts.generatorType](os.path.join(opts.dataDir,'train', opts.dataType),
														 opts.ext, opts.batchSize, None, mode='train')
	val_generator = generators_dict[opts.generatorType](os.path.join(opts.dataDir,'val', opts.dataType), 
														opts.ext, opts.batchSize*2, None, mode='validation')
														
	steps_per_epoch = len(os.listdir(os.path.join(opts.dataDir,'train', opts.dataType, 'X'))) / opts.batchSize
	validation_steps = len(os.listdir(os.path.join(opts.dataDir,'val', opts.dataType, 'X'))) / (opts.batchSize*2)

	#Configuring experimentation directories,callbacks and stuff..
	expDir = os.path.join(opts.logRootDir, opts.logDir)
	if not os.path.exists(expDir): 
		os.makedirs(expDir)
		writeConfigToFile(os.path.join(expDir,'opts.txt'), vars(opts), model)
		
	os.makedirs(os.path.join(expDir, 'model'))
	ckptCallback=ModelCheckpoint(os.path.join(expDir,'model', '{epoch:02d}-{loss:.2f}.hdf5'),
								monitor='loss',save_best_only=True)

	return 
	#FINALLY! TRAINING NOW..
	history = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=opts.numEpochs
								,verbose=opts.verbosity, validation_data=val_generator, validation_steps=validation_steps)
	return


def SetArguments(parser): 
	#Data loading arguments
	parser.add_argument('-dataDir',action='store', type=str, default='../data/generated2/', dest='dataDir')
	parser.add_argument('-dataType',action='store', type=str, default='noNoise', dest='dataType')
	parser.add_argument('-ext', action='store',type=list, default=['png', 'jpg'], dest='ext')
	parser.add_argument('-generatorType', action='store', type=str, default='generator_full_image', dest='generatorType')
	parser.add_argument('-inputShape', action='store', type=tuple, default=(512,512,1), dest='inputShape')

	#Model parameters
	parser.add_argument('-netType', action='store', type=str, default='imageToImageSeq', dest='netType')
	parser.add_argument('-dropRate', action='store', type=float, default=0.0, dest='dropRate')
	parser.add_argument('-kernelSizes', action='store', type=str, default='3,3,1', dest='kernelSizes')
	parser.add_argument('-numKernels', action='store', type=str, default='32,32,1', dest='numKernels')
	parser.add_argument('-activations', action='store', type=str, default='relu,relu,sigmoid', dest='activations')
	parser.add_argument('-padding', action='store', type=str, default='same', dest='padding')
	parser.add_argument('-strides', action='store', type=int, default=1, dest='strides')

	#Learning parameters
	parser.add_argument('-optimizerType', action='store', type=str, default='adam', dest='optimizerType')
	parser.add_argument('-learningRate', action='store', type=float, default=1e-3, dest='learningRate')
	parser.add_argument('-lrDecay', action='store', type=float, default=0.0, dest='lrDecay')
	parser.add_argument('-numEpochs', action='store', type=int, default=1, dest='numEpochs')
	parser.add_argument('-verbosity', action='store', type=int, default=1, dest='verbosity')
	parser.add_argument('-batchSize', action='store', type=int, default=1024, dest='batchSize')

	#Loss function parameters
	parser.add_argument('-lossType', action='store', type=str, default='binary_crossentropy', dest='lossType')

	#Logging parameters
	parser.add_argument('-logRootDir',action='store',type=str, default='../experiments/',dest='logRootDir')
	parser.add_argument('-logDir',action='store',type=str, default=strftime("%d-%m-%Y__%H-%M-%S",gmtime()),dest='logDir')
	return

def PostprocessOpts(opts): 
	opts.kernelSizes = [int(x) for x in opts.kernelSizes.split(',')]
	opts.numKernels = [int(x) for x in opts.numKernels.split(',')]
	opts.activations = opts.activations.split(',')
	return 

if __name__=='__main__': 
	parser = argparse.ArgumentParser()
	SetArguments(parser)

	opts = parser.parse_args()
	PostprocessOpts(opts)

	train(opts)