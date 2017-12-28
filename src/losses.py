import keras.backend as K

def BinaryCrossEntropy(y_true, y_pred):	
	losses = -((y_true*K.log(y_pred)) + ((1-y_true)*K.log(1-y_pred)))
	return K.mean(losses)

def WeightedBinaryCrossEntropy(y_true, y_pred):
	err = -((y_true*K.log(y_pred)) + ((1-y_true)*K.log(1-y_pred)))
	probs = K.mean(y_true,axis=(1,2,3),keepdims=True)
	weights = 1./probs
	return K.mean(err*weights)

def L2Loss(y_true,y_pred): 
	return K.mean(K.square(y_true-y_pred))

def WeightedL2Loss(y_true,y_pred): 
	err = K.square(y_true-y_pred)
	probs = K.mean(y_true,axis=(1,2,3),keepdims=True)
	weights = 1./probs
	return K.mean(weights*err)

losses_dict={'binaryCrossEntropy':BinaryCrossEntropy,'weightedBinaryCrossEntropy':WeightedBinaryCrossEntropy, 
			'l2Loss': L2Loss, 'weightedL2Loss':WeightedL2Loss}