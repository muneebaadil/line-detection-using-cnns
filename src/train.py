import argparse
import numpy as np
import keras as krs
import importlib
from models import model_dummy #hardcoding imported nettype for now.. 

def train(opts): 
	"""Performs the whole algorithm i.e trains a given neural network on given data using given learning parameters
	
	Args: 
	opts: command line arguments
	
	Returns: 
	None"""

	#Creating given model
	model = model_dummy.create_net(opts)

	#Compiling given model using given learning parameters.
	pass

def set_arguments(parser): 
	pass

if __name__=='__main__': 
	parser = argparse.ArgumentParser()
	set_arguments(parser)

	opts = parser.parse_args()

	train(opts)