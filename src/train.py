import argparse
import numpy as np
import importlib
import os 

from models import models_dict
from losses import losses_dict 
from optimizers import optimizers_dict
from generators import generators_dict

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
	model.compile(optimizer=optimizer, loss=losses_dict[opts.lossType])

	#Setting up callbacks..
	#return 
	
	#Configuring data loaders/generators now..
	train_generator = generators_dict[opts.generatorType](os.path.join(opts.dataDir,'train'), opts.ext, opts.batchSize, None, mode='train')
	val_generator = generators_dict[opts.generatorType](os.path.join(opts.dataDir,'val'), opts.ext, opts.batchSize*2, None, mode='validation')
	steps_per_epoch = 1
	validation_steps = 1 

	#FINALLY! TRAINING NOW..
	history = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=opts.numEpochs
								,verbose=opts.verbosity, validation_data=val_generator, validation_steps=validation_steps)
	return


def set_arguments(parser): 
	#Data loading arguments
	parser.add_argument('-dataDir',action='store', type=str, default='../data/prepared/', dest='dataDir')
	parser.add_argument('-ext', action='store',type=list, default=['png', 'jpg'], dest='ext')
	parser.add_argument('-generatorType', action='store', type=str, default='generator_full_image', dest='generatorType')

	#Model parameters
	parser.add_argument('-netType', action='store', type=str, default='model_init', dest='netType')

	#Learning parameters
	parser.add_argument('-optimizerType', action='store', type=str, default='adam', dest='optimizerType')
	parser.add_argument('-learningRate', action='store', type=float, default=1e-3, dest='learningRate')
	parser.add_argument('-lrDecay', action='store', type=float, default=0.0, dest='lrDecay')

	#Loss function parameters
	parser.add_argument('-lossType', action='store', type=str, default='binary_crossentropy', dest='lossType')
	pass 

if __name__=='__main__': 
	parser = argparse.ArgumentParser()
	set_arguments(parser)

	opts = parser.parse_args()

	train(opts)