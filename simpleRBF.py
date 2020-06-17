from inputLayer import inputLayer
from rbfLayer import rbfLayer
from outputLayer import outputLayer
import numpy as np

class RBF:
    def __init__(self,inputLayer,RBFLayer,outputLayer):
        self.inputLayer = inputLayer
        self.RBFLayer = RBFLayer
        self.outputLayer = outputLayer
    
    def calculateWeights(self,input, trainOutput):
        matrixM = self.RBFLayer.outputMatrix(input)
        # invert matrix
        matrixM = np.linalg.inv(matrixM)
        self.weights = matrixM.dot(matrixM,trainOutput)



if __name__ == "__main__":
    centers = np.arange(0.5,4.6,1)
    width = 0.7
    inputs = np.arange(0,6,1)
    trainings_outputs = np.sin(inputs)

    myInputLayer = inputLayer(1)
    myrbfLayer = rbfLayer(centers, width)
    myOutputLayer = outputLayer(1)
