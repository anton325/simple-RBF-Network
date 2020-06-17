from inputLayer import inputLayer
from rbfLayer import RBFLayer
from outputLayer import outputLayer
import numpy as np

class RBF:
    def __init__(self,inputLayer,RBFLayer,outputLayer):
        self.inputLayer = inputLayer
        self.RBFLayer = RBFLayer
        self.outputLayer = outputLayer
    
    def calculateWeights(self,input, trainOutput):
        matrixM = self.RBFLayer.outputMatrix(input)
        print("matrixM hat vor invertieren shape: "+str(matrixM.shape))
        # invert matrix
        matrixM = np.linalg.pinv(matrixM) # get pseudoinverse
        print("matrixM hat NACH invertieren shape: "+str(matrixM.shape))

        print("matrixM: "+str(matrixM)+" training outputs: "+str(trainings_outputs))
        print("Training outputs hat die shape: "+str(trainings_outputs.shape))
        weights = np.dot(matrixM,trainOutput)
        print("berchnet wurden die gewichte: "+str(weights))

        # tell output layer
        self.outputLayer.introduceWeightMatrix(weights)


    def guessOutput(self,scalarInput):
        outputMatrix = self.RBFLayer.outputToScalar(scalarInput)
        res = self.outputLayer.calculateOutput(outputMatrix)
        return(res)




if __name__ == "__main__":
    centers = np.arange(0.5,5.6,1)
    width = 0.7
    inputs = np.arange(0,6,0.5)
    trainings_outputs = np.sin(inputs)

    myInputLayer = inputLayer(1)
    myrbfLayer = RBFLayer(centers, width)
    myOutputLayer = outputLayer(1)

    myRbf = RBF(myInputLayer,myrbfLayer,myOutputLayer)
    myRbf.calculateWeights(inputs,trainings_outputs)


    correct = np.array([])
    guessed = np.array([])
    for testInput in np.arange(0,5,0.1):
        print("Zum Input: "+str(testInput)+" berechnet das Netz: "+str(myRbf.guessOutput(testInput))+ " erwartet wurde: "+str(np.sin(testInput)))
        correct = np.append(correct, np.sin(testInput))
        guessed = np.append(guessed, myRbf.guessOutput(testInput))

    print(correct)
    print(guessed)
