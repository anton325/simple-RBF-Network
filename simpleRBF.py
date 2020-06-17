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
        print("Gradient descent took: "+str(t2-t1))




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

def saveWeightsAndCenters(weights, centers):
    np.savez("rbf.npz",name1 = weights, name2 = centers)
    print("weights saved")

def plotWithoutGrad(x):
    correct = np.array([])
    guessed = np.array([])
    usedRange = x
    for testInput in usedRange:
        guessed = np.append(guessed, myRbf.guessOutput(testInput))
        correct = np.append(correct, wishFunction(testInput))
    
    # show input neurons
    myInputLayer.plotNeurons()
    
    plt.plot(usedRange,correct, color = "g")
    plt.plot(usedRange,guessed, color = "r")
    #plt.show()
    return guessed

def plotWithGrad(x,oldGuess):
    correct = np.array([])
    newGuessed = np.array([])
    usedRange = x
    for testInput in usedRange:
        newGuessed = np.append(newGuessed, myRbf.guessOutput(testInput))
        correct = np.append(correct, wishFunction(testInput))
    
    # show input neurons
    myInputLayer.plotNeurons()
    
    plt.plot(usedRange,oldGuess, color = "r")
    plt.plot(usedRange,correct, color = "g")
    plt.plot(usedRange,newGuessed, color = "b")
    plt.show()

def performeGrad(rbf,numEpoch, usedRange):
    rbf.gradientDescent(usedRange, wishFunction(usedRange), numEpochs)
    return rbf.weightsWithGradient



if __name__ == "__main__":
    xmax = 20

    # Distribute centers evenly
    distanceBetween = 0.7
    centers = np.arange(0,xmax,distanceBetween)
    width = distanceBetween*0.6

    # Determine with how many values you want to estimate first set of weights
    inputs = np.arange(0,xmax,1)
    trainings_outputs = wishFunction(inputs)

    # define single layers
    myInputLayer = inputLayer(1,centers,width)
    myrbfLayer = RBFLayer(centers, width)
    myOutputLayer = outputLayer(1)

    # get model and calculate weights
    myRbf = RBF(myInputLayer,myrbfLayer,myOutputLayer)
    myRbf.calculateWeights(inputs,trainings_outputs)

    # plot initial fit
    plotRange =np.arange(0,xmax,0.01)
    oldGuess = plotWithoutGrad(plotRange)

    # grad descent
    numEpochs = 10
    gradRange = np.arange(0,xmax,0.25)
    newWeights = performeGrad(myRbf, numEpochs, gradRange)

    # save calculated weights
    saveWeightsAndCenters(newWeights,centers)

    # plot with new AND old weights
    plotWithGrad(plotRange,oldGuess)
   


