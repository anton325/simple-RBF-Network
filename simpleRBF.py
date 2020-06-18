from inputLayer import inputLayer
from rbfLayer import RBFLayer
from outputLayer import outputLayer
import numpy as np
import matplotlib.pyplot as plt
import time

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

        #print("matrixM: "+str(matrixM)+" training outputs: "+str(trainings_outputs))
        #print("Training outputs hat die shape: "+str(trainings_outputs.shape))
        self.weightsWithoutGradient = np.dot(matrixM,trainOutput)
        print("berchnet wurden die gewichte: "+str(self.weightsWithoutGradient))

        # tell output layer
        self.outputLayer.introduceWeightMatrix(self.weightsWithoutGradient)


    """
    Obtain output of rbf network to given scalar input
    """
    def guessOutput(self,scalarInput):
        """
        guessOutput
        recieve single scalar value. Obtain output matrix from rbf layer 
        And multiply with weights to get guess
        """
        outputMatrix = self.RBFLayer.outputToScalar(scalarInput)
        res = self.outputLayer.calculateOutput(outputMatrix)
        return(res)



    def gradientDescent(self, teachingInput, teachingOutput, epoch):
        #start timer
        t1 = time.time()


        # lernrate zwischen 0.01 und 0.9
        print("gradient descent")
        counter = 0
        self.weightsWithGradient = self.weightsWithoutGradient

        # calculate delta
        speed = 1
        while counter < epoch:
            speed = 0.9 * speed
            for i in range(np.size(teachingInput)):
                for neuron in range(self.RBFLayer.numberOfNeurons):
                    # calculate activation:
                    center = self.RBFLayer.centers[neuron]
                    eingang =  teachingInput[i]
                    activation = self.RBFLayer.getOutputOfSingleNeuron(center , eingang)
                    print("Prüfe in Durchlauf: "+str(counter)+" von "+str(epoch)+ " für Eingang: "+str(eingang))
                    # what would the network usually guess?
                    guess = self.guessOutput(teachingInput[i])
                    print("guess would have been: "+str(guess)+ " it should have been: "+str(teachingOutput[i]))
                    # calculate difference
                    diff = teachingOutput[i]-guess
                    # calculate gradient
                    grad = speed*diff*activation

                    print("Zum Input: "+str(teachingInput[i])+" ist der Unterschied: "+str(diff)+" grad: "+str(grad))
                    # update the respective weight:
                    self.weightsWithGradient[neuron] = self.weightsWithGradient[neuron]+grad
                i = i + 1

            # increment counter
            counter = counter + 1

        # tell output layer about new weights
        self.outputLayer.introduceWeightMatrix(self.weightsWithGradient)   
        t2 = time.time()
        print("Gradient descent took: "+str(t2-t1)+" s")




def wishFunction(x):
    """
    wishFunction
    determine here which function you want to realise
    """
    if not (type(x) == type(np.array([]))):
        #print("kein array")
        return wishFunction(np.array([x]))
    y = np.array([])
    for values in x: 
        y = np.append(y,
                    2*np.sin(values) -3*np.cos(3*values) +np.exp(-values*values) -2.3*np.sin(3*values))
    return y

def saveConfig(model, distance, width):
    neurons = np.array([distance,width])
    np.savez("rbf.npz",name1 = model.outputLayer.weightMatrix, name2 = model.RBFLayer.centers, name3 = neurons)
    print("config saved")

def loadModel():
    # get data
    data = np.load("rbf.npz")
    weights = data["name1"]
    centers = data["name2"]
    neurons = data["name3"]
    distanceBetween = neurons[0]
    width = neurons[1]
    # get model
    loadedModel = getModel(1,centers,width,1)
    # train model
    loadedModel.outputLayer.introduceWeightMatrix(weights)
    return loadedModel

def plotWithoutGrad(myRbf,x):
    correct = np.array([])
    guessed = np.array([])
    usedRange = x
    for testInput in usedRange:
        guessed = np.append(guessed, myRbf.guessOutput(testInput))
        correct = np.append(correct, wishFunction(testInput))
    
    # show input neurons
    myRbf.RBFLayer.plotNeurons()
    
    plt.plot(usedRange,correct, color = "g")
    plt.plot(usedRange,guessed, color = "r")
    #plt.show()
    return guessed

def plotWithGrad(myRbf,x,oldGuess):
    correct = np.array([])
    newGuessed = np.array([])
    usedRange = x
    for testInput in usedRange:
        newGuessed = np.append(newGuessed, myRbf.guessOutput(testInput))
        correct = np.append(correct, wishFunction(testInput))
    
    # show input neurons
    myRbf.RBFLayer.plotNeurons()
    
    plt.plot(usedRange,oldGuess, color = "r")
    plt.plot(usedRange,correct, color = "g")
    plt.plot(usedRange,newGuessed, color = "b")
    plt.show()


def performeGrad(rbf,numEpochs, usedRange):
    # performe grad
    rbf.gradientDescent(usedRange, wishFunction(usedRange), numEpochs)
    # save new weights
    rbf.outputLayer.introduceWeightMatrix(rbf.weightsWithGradient)

def getModel(numInputNeurons,centers,width,numOutputNeurons):
     # define single layers
    myInputLayer = inputLayer(1)
    myrbfLayer = RBFLayer(centers, width)
    myOutputLayer = outputLayer(1)

    # get model
    return(RBF(myInputLayer,myrbfLayer,myOutputLayer))


def startFromScrap(xMax,epochs):
    # Determine with how many values you want to estimate first set of weights
    xmax = xMax
    inputs = np.arange(0,xmax,1)
    trainings_outputs = wishFunction(inputs)

    # Distribute centers evenly
    distanceBetween = 0.7
    centers = np.arange(0,xmax,distanceBetween)
    width = distanceBetween*0.6
    # get Model
    myRbf = getModel(1,centers,width,1)

    

    # get weights
    myRbf.calculateWeights(inputs,trainings_outputs)

    # plot initial fit with only the estimated weights
    plotRange =np.arange(0,xmax,0.01)
    oldGuess = plotWithoutGrad(myRbf,plotRange)

    # Performe grad descent
    numEpochs = epochs
    gradRange = np.arange(0,xmax,0.25)
    performeGrad(myRbf, numEpochs, gradRange)

    # save calculated weights
    saveConfig(myRbf, distanceBetween,width)

     # plot with new AND old weights
    plotWithGrad(myRbf,plotRange,oldGuess)




if __name__ == "__main__":
    startFromScrap(20,10)

    # load weights and centers
    #weights,centers = loadWeights()
    #myRbf.outputLayer.introduceWeightMatrix(weights)

   
   


