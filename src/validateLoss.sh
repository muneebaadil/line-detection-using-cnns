python train.py -dataDir ../data/generated/ -numEpochs 10 -batchSize 16 -lossType binaryCrossEntropy -logPerEpoch 5
python train.py -dataDir ../data/generated/ -numEpochs 10 -batchSize 16 -lossType WeightedBinaryCrossEntropy -logPerEpoch 5

python train.py -dataDir ../data/generated/ -kernelSizes 3,3,3,1 -numKernels 32,32,32,1 -activations relu,relu,relu,sigmoid -numEpochs 10 -batchSize 16 -lossType binaryCrossEntropy -logPerEpoch 5
python train.py -dataDir ../data/generated/ -kernelSizes 3,3,3,1 -numKernels 32,32,32,1 -activations relu,relu,relu,sigmoid -numEpochs 10 -batchSize 16 -lossType WeightedBinaryCrossEntropy -logPerEpoch 5