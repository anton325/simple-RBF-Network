from inputLayer import inputLayer
from rbfLayer import RBFLayer
from outputLayer import outputLayer
import numpy as np
import matplotlib.pyplot as plt

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

def wishFunction(x):
    y = np.array([])
    for values in x: 
        y = np.append(y,2*np.sin(values)-3*np.cos(3*values)+np.exp(-values*values))
    return y



if __name__ == "__main__":
    # Distribute centers evenly
    centers = np.arange(0.1,5.6,0.3)
    width = 0.24

    # Determine with how many values you want to learn
    inputs = np.arange(0,6,0.01)
    trainings_outputs = wishFunction(inputs)
    print("trainings outputs are: asdf " + str(trainings_outputs))

    myInputLayer = inputLayer(1)
    myrbfLayer = RBFLayer(centers, width)
    myOutputLayer = outputLayer(1)

    myRbf = RBF(myInputLayer,myrbfLayer,myOutputLayer)
    myRbf.calculateWeights(inputs,trainings_outputs)


    correct = np.array([])
    guessed = np.array([])
    print("start gathering values")
    for testInput in np.arange(0,5,0.1):
        #print("Zum Input: "+str(testInput)+" berechnet das Netz: "+str(myRbf.guessOutput(testInput))+ " erwartet wurde: "+str(np.sin(testInput)))
        guessed = np.append(guessed, myRbf.guessOutput(testInput))
    
    a = np.arange(0,5,0.1,dtype = float)
    correct = wishFunction(a)

    plt.plot(correct, color = "g")
    plt.plot(guessed, color = "r")
    plt.show()
