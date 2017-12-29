import argparse
import numpy as np
import importlib
import os 
from time import gmtime, strftime
from keras.callbacks import ModelCheckpoint, TensorBoard
import pickle
from keras.models import load_model

from models import models_dict
from optimizers import optimizers_dict
from generators import generators_dict
from losses import losses_dict
from utils import * 

import pdb 

def train(opts): 
	"""Performs the whole algorithm i.e trains a given neural network on given data using given learning parameters
	
	Args: 
	opts: command line arguments
	
	Returns: 
	None"""

	#Creating given model OR loading pretrained network..
	if opts.loadModel is None: 
		print 'Creating Network Architecture..'
		model = models_dict[opts.netType](opts)
	else: 
		print 'Loading Pretrained Network from {}..'.format(opts.loadModel)
		model = load_model(opts.loadModel,compile=False)

	#Compiling given model using given learning parameters..
	optimizer = optimizers_dict[opts.optimizerType](lr=opts.learningRate, decay=opts.lrDecay)
	model.compile(optimizer=optimizer, loss=losses_dict[opts.lossType](model.input))

	#Configuring data loaders/generators now..
	train_generator = generators_dict[opts.generatorType](os.path.join(opts.dataDir,'train', opts.dataType),
														 opts.ext, opts.batchSize, None, mode='train')
	val_generator = generators_dict[opts.generatorType](os.path.join(opts.dataDir,'val', opts.dataType), 
														opts.ext, opts.batchSize*2, None, mode='validation')
														
	steps_per_epoch = (len(os.listdir(os.path.join(opts.dataDir,'train', opts.dataType, 'X'))) / opts.batchSize) / opts.logPerEpoch
	validation_steps = len(os.listdir(os.path.join(opts.dataDir,'val', opts.dataType, 'X'))) / (opts.batchSize*2)
	numEpochs_ = opts.numEpochs * opts.logPerEpoch

	#Configuring experimentation directories..
	if not os.path.exists(opts.expDir): 
		os.makedirs(opts.expDir)
		writeConfigToFile(os.path.join(opts.expDir,'opts.txt'), vars(opts), model)
	
	#Configuring callbacks..
	os.makedirs(os.path.join(opts.expDir, 'model'))
	ckptCallback=ModelCheckpoint(os.path.join(opts.expDir,'model', '{epoch:02d}-{loss:.2f}.hdf5'),
								monitor='loss',save_best_only=True)
	tboardCallback=TensorBoard(log_dir=os.path.join(opts.expDir,'tensorboardLogs'),histogram_freq=1,
								batch_size=opts.batch_size,write_graph=False, write_grads=True)
	valsaver = valImagesSaver(dataDir=os.path.join(opts.dataDir,'val', opts.dataType, 'X'),
					ext=opts.ext, outDir=os.path.join(opts.expDir, 'valImages'))

	#FINALLY! TRAINING NOW..
	history = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=numEpochs_,
								verbose=opts.verbosity, validation_data=val_generator, validation_steps=validation_steps,
								callbacks=[ckptCallback,tboardCallback,valsaver])

	with open(os.path.join(opts.expDir, 'trainHistory'), 'wb') as fobj: 
		pickle.dump(history.history, fobj)
	return


def SetArguments(parser): 
	#Data loading arguments
	parser.add_argument('-dataDir',action='store', type=str, default='../data/generated/', dest='dataDir')
	parser.add_argument('-dataType',action='store', type=str, default='noNoise', dest='dataType')
	parser.add_argument('-ext', action='store',type=list, default=['png', 'jpg'], dest='ext')
	parser.add_argument('-generatorType', action='store', type=str, default='generator_full_image', dest='generatorType')
	parser.add_argument('-inputShape', action='store', type=tuple, default=(512,512,1), dest='inputShape')

	#Model parameters
	parser.add_argument('-netType', action='store', type=str, default='imageToImageSeq', dest='netType')
	parser.add_argument('-dropRate', action='store', type=float, default=0.0, dest='dropRate')
	parser.add_argument('-kernelSizes', action='store', type=str, default='3,3,3', dest='kernelSizes')
	parser.add_argument('-numKernels', action='store', type=str, default='16,16,1', dest='numKernels')
	parser.add_argument('-activations', action='store', type=str, default='relu,relu,sigmoid', dest='activations')
	parser.add_argument('-padding', action='store', type=str, default='same', dest='padding')
	parser.add_argument('-strides', action='store', type=int, default=1, dest='strides')
	parser.add_argument('-includeInsNormLayer', action='store', type=bool, default=False, dest='includeInsNormLayer')
	parser.add_argument('-insNormAxis', action='store', type=int, default=None, dest='insNormAxis')
	parser.add_argument('-loadModel', action='store', type=str, default=None, dest='loadModel')

	#Learning parameters
	parser.add_argument('-optimizerType', action='store', type=str, default='adam', dest='optimizerType')
	parser.add_argument('-learningRate', action='store', type=float, default=1e-3, dest='learningRate')
	parser.add_argument('-lrDecay', action='store', type=float, default=0.0, dest='lrDecay')
	parser.add_argument('-numEpochs', action='store', type=int, default=1, dest='numEpochs')
	parser.add_argument('-verbosity', action='store', type=int, default=1, dest='verbosity')
	parser.add_argument('-batchSize', action='store', type=int, default=16, dest='batchSize')

	#Loss function parameters
	parser.add_argument('-lossType', action='store', type=str, default='weightedBinaryCrossEntropy', dest='lossType')

	#Logging parameters
	parser.add_argument('-logRootDir',action='store',type=str, default='../experiments/',dest='logRootDir')
	parser.add_argument('-logDir',action='store',type=str, default=strftime("%d-%m-%Y__%H-%M-%S",gmtime()),dest='logDir')
	parser.add_argument('-logPerEpoch',action='store',type=int, default=1,dest='logPerEpoch')
	return

def PostprocessOpts(opts): 
	opts.kernelSizes = [int(x) for x in opts.kernelSizes.split(',')]
	opts.numKernels = [int(x) for x in opts.numKernels.split(',')]
	opts.activations = opts.activations.split(',')
	opts.expDir = os.path.join(opts.logRootDir, opts.logDir)
	return 

if __name__=='__main__': 
	parser = argparse.ArgumentParser()
	SetArguments(parser)

	opts = parser.parse_args()
	PostprocessOpts(opts)

	train(opts)
