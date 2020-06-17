from inputLayer import inputLayer
from rbfLayer import RBFLayer
from outputLayer import outputLayer
import numpy as np
import matplotlib.pyplot as plt

class RBF:
    """
    __ini__
    Übergebe definierte Layers
    """
        
    def __init__(self,inputLayer,RBFLayer,outputLayer):
         
        self.inputLayer = inputLayer
        self.RBFLayer = RBFLayer
        self.outputLayer = outputLayer
    
    
    def calculateWeights(self,input, trainOutput):
        """
        calculateWeights
        calculate the weights by getting the M matrix, obtaining the pseudo inverse and multiplying with training output
        """
        matrixM = self.RBFLayer.outputMatrix(input)
        print("matrixM hat vor invertieren shape: "+str(matrixM.shape))
        # invert matrix
        matrixM = np.linalg.pinv(matrixM) # get pseudoinverse
        print("matrixM hat NACH invertieren shape: "+str(matrixM.shape))

        print("matrixM: "+str(matrixM)+" training outputs: "+str(trainings_outputs))
        print("Training outputs hat die shape: "+str(trainings_outputs.shape))
        self.weightsWithoutGradient = np.dot(matrixM,trainOutput)
        print("berchnet wurden die gewichte: "+str(self.weightsWithoutGradient))

        # tell output layer
        self.outputLayer.introduceWeightMatrix(self.weightsWithoutGradient)


    
    def guessOutput(self,scalarInput):
        """
        guessOutput
        recieve single scalar value. Obtain output matrix from rbf layer 
        And multiply with weights to get guess
        """
        outputMatrix = self.RBFLayer.outputToScalar(scalarInput)
        res = self.outputLayer.calculateOutput(outputMatrix)
        return(res)


def wishFunction(x):
    """
    wishFunction
    determine here which function you want to realise
    """

    if not (type(x) == type(np.array([]))):
        print("kein array")
        return wishFunction(np.array([x]))
        

    y = np.array([])
    for values in x: 
        y = np.append(y,2*np.sin(values)-3*np.cos(3*values)+np.exp(-values*values))
    return y



if __name__ == "__main__":
    # Distribute centers evenly
    distanceBetween = 1
    centers = np.arange(0,20,distanceBetween)
    width = distanceBetween*0.6

    # Determine with how many values you want to learn
    inputs = np.arange(0,20,0.1)
    trainings_outputs = wishFunction(inputs)
    print("trainings outputs are: asdf " + str(trainings_outputs))

    # define single layers
    myInputLayer = inputLayer(1,centers,width)
    myrbfLayer = RBFLayer(centers, width)
    myOutputLayer = outputLayer(1)

    

    # get model and calculate weights
    myRbf = RBF(myInputLayer,myrbfLayer,myOutputLayer)
    myRbf.calculateWeights(inputs,trainings_outputs)


    print("start gathering values")
    correct = np.array([])
    guessed = np.array([])
    usedRange = np.arange(0,14,0.1)
    for testInput in usedRange:
        #print("Zum Input: "+str(testInput)+" berechnet das Netz: "+str(myRbf.guessOutput(testInput))+ " erwartet wurde: "+str(np.sin(testInput)))
        guessed = np.append(guessed, myRbf.guessOutput(testInput))
        correct = np.append(correct, wishFunction(testInput))
    
    #a = np.arange(0,5,0.1,dtype = float)
    #correct = wishFunction(a)

    # show input neurons
    myInputLayer.plotNeurons()
    
    plt.plot(usedRange,correct, color = "g")
    plt.plot(usedRange,guessed, color = "r")
    plt.show()
