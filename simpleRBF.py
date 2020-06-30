from inputLayer import inputLayer
from rbfLayer import RBFLayer
from outputLayer import outputLayer
import numpy as np
import matplotlib.pyplot as plt
import time
import os

class RBF:
   
        
    def __init__(self,inputLayer,RBFLayer,outputLayer):
        """
        __init__
        Save already defined layers
        """
        self.inputLayer = inputLayer
        self.RBFLayer = RBFLayer
        self.outputLayer = outputLayer
    
  
    def calculateWeights(self,input, trainOutput):
        """
        calculateWeights
        calculate the weights by getting the M matrix from the rbflayer, obtaining the pseudo inverse and multiplying with training output (the desired output)
        """

        matrixM = self.RBFLayer.outputMatrix(input)

        matrixM = np.linalg.pinv(matrixM) # get pseudoinverse

        self.weightsWithoutGradient = np.dot(matrixM,trainOutput) # multiply M with training output
        print("we calculated the weights: \n"+str(self.weightsWithoutGradient))

        # tell output layer
        self.outputLayer.introduceWeightMatrix(self.weightsWithoutGradient)



    def guessOutput(self,scalarInput):
        """
        guessOutput
        recieve single scalar value. Obtain output matrix from rbf layer 
        And multiply with weights to get guess
        """
        outputMatrix = self.RBFLayer.outputToScalar(scalarInput)
        result = self.outputLayer.calculateOutput(outputMatrix)
        return result



    def gradientDescent(self, teachingInput, teachingOutput, epoch):
        """
        gradientDescent
        perform a gradient descent algorithm. In each epoch calculate the differences between output of the network and the desired output
        Adjust the weights accordingly
        """
        #start timer
        t1 = time.time()

        # speed between 0.01 and 0.9 (starting at 0.9 and decreasing)
        speed = 0.9

        print("perform gradient descent")
        counter = 0
        self.weightsWithGradient = self.weightsWithoutGradient

        while counter < epoch:
            for i in range(np.size(teachingInput)):
                # for each input
                for neuron in range(self.RBFLayer.numberOfNeurons):
                    # for each neuron

                    # calculate activation of this single neuron:
                    center = self.RBFLayer.centers[neuron]
                    input =  teachingInput[i]
                    activation = self.RBFLayer.getOutputOfSingleNeuron(center , input)
                    print("Testing in iteration: "+str(counter+1)+" of "+str(epoch)+ " input: "+str(input))

                    # what would the network usually guess?
                    guess = self.guessOutput(teachingInput[i])
                    print("guess would have been: "+str(guess)+ " it should have been: "+str(teachingOutput[i]))

                    # calculate difference
                    diff = teachingOutput[i]-guess
                    # calculate gradient
                    grad = speed*diff*activation

                    print("For input: "+str(teachingInput[i])+" the difference is: "+str(diff)+" and the gradient is: "+str(grad))
                    # update the respective weight:
                    self.weightsWithGradient[neuron] = self.weightsWithGradient[neuron]+grad
            #tell outputlayer new weights
            self.outputLayer.introduceWeightMatrix(self.weightsWithGradient)
            # increment counter
            counter = counter + 1
            # adjust speed
            speed = 0.95 * speed


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
        #print("Input is not an array (as it should be)")
        return wishFunction(np.array([x]))
    y = np.array([])
    for values in x: 
        y = np.append(y,
                    2*np.sin(values) -3*np.cos(3*values) +np.exp(-values*values) -2.3*np.sin(3*values))
                    #np.exp(-(values-4)*(values-4))+np.exp(-3*(values-1))+np.log(values))
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

def plotWithGrad(myRbf,x,oldGuess,savePlot,epochs):
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

    if savePlot == True:
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "Plots/onlineLearning3/Epochs "+str(epochs)+".png"
        abs_file_path = os.path.join(script_dir, rel_path)      
        plt.savefig(abs_file_path)
        print("Plot saved")
    else:
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


def startFromScrap(xMax,epochs,savePlot):
    # Determine with how many values you want to estimate first set of weights
    xmax = xMax
    inputs = np.arange(0.5,xmax,0.5)
    trainings_outputs = wishFunction(inputs)

    # Distribute centers evenly
    distanceBetween = 1
    centers = np.arange(0,xmax,distanceBetween)
    width = distanceBetween*0.65
    # get Model
    myRbf = getModel(1,centers,width,1)

    

    # get weights
    myRbf.calculateWeights(inputs,trainings_outputs)

    # plot initial fit with only the estimated weights
    plotRange =np.arange(0.5,xmax,0.01)
    oldGuess = plotWithoutGrad(myRbf,plotRange)

    # Performe grad descent
    numEpochs = epochs
    gradRange = np.arange(0.1,xmax,0.1)
    performeGrad(myRbf, numEpochs, gradRange)

    # save calculated weights
    #saveConfig(myRbf, distanceBetween,width)

     # plot with new AND old weights
    plotWithGrad(myRbf,plotRange,oldGuess,savePlot,numEpochs)





if __name__ == "__main__":
    #startFromScrap(14,1,True)
    # get plots with respect to numEpochs
    #for epoch in range(30):

    # startFromScrap(xMax, epochs, saveplot)
    startFromScrap(12,8,False)
    #plt.plot(wishFunction(np.arange(0.5,15,0.5)))
    #plt.show()
    # load weights and centers
    #weights,centers = loadWeights()
    #myRbf.outputLayer.introduceWeightMatrix(weights)

   
   


