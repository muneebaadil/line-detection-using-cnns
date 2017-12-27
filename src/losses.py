import keras.backend as K

def BinaryCrossEntropy(y_true, y_pred):	
	losses = -((y_true*K.log(y_pred)) + ((1-y_true)*K.log(1-y_pred)))
	return K.mean(losses)

def WeightedBinaryCrossEntropy(y_true, y_pred):
	err = -((y_true*K.log(y_pred)) + ((1-y_true)*K.log(1-y_pred)))
	probs = K.mean(y_pred,axis=(1,2,3),keepdims=True)
	weights = 1./probs
	return K.mean(err*weights)

losses_dict={'binaryCrossEntropy':BinaryCrossEntropy,'weightedBinaryCrossEntropy':WeightedBinaryCrossEntropy}