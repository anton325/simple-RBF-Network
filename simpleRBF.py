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
        __ini__
        Pre-defined layers as parameters that just have to be saved
        """
        self.inputLayer = inputLayer
        self.RBFLayer = RBFLayer
        self.outputLayer = outputLayer
    
    
    def calculateWeights(self,input, trainOutput):
        """
        calculateWeights
        calculate the weights by getting the M matrix, obtaining the pseudo inverse and multiplying
        with training output
        """
        matrixM = self.RBFLayer.outputMatrix(input)
        # invert matrix
        matrixM = np.linalg.pinv(matrixM) # get pseudoinverse

        self.weightsWithoutGradient = np.dot(matrixM,trainOutput)
        print("Calculated the weights:\n"+str(self.weightsWithoutGradient))

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
        return(result)



    def gradientDescent(self, teachingInput, teachingOutput, epoch, saveAfterEachEpoch):
        """
        perform the gradient descent algorithm
        @teachingInput: The training input
        @teachingOutput: The training output (desired output to training input)
        @epoch: How many times to repeat the whole process
        @saveAfterEachEpoch: If true plot is saved after each successful epoch
        """
        # save teaching in and output
        origInput = np.copy(teachingInput)
        origOutput = np.copy(teachingOutput)

        #start timer
        t1 = time.time()

        # speed of algo between 0.01 and 0.9
        speed = 0.9

        print("Performing gradient descent...")
        counter = 0
        self.weightsWithGradient = self.weightsWithoutGradient

        # calculate delta and refresh weights
        while counter < epoch:
            for i in range(np.size(teachingInput)):
                for neuron in range(self.RBFLayer.numberOfNeurons):
                    # calculate activation of the single neuron:
                    center = self.RBFLayer.centers[neuron]
                    trainingInput =  teachingInput[i]
                    activation = self.RBFLayer.getOutputOfSingleNeuron(center , trainingInput)
                    #print("Test in iteration: "+str(counter+1)+" of "+str(epoch)+ " for input: "+str(trainingInput))

                    # what would the network usually guess?
                    guess = self.guessOutput(teachingInput[i])
                    #print("Guess would have been: "+str(guess)+ " it should have been: "+str(teachingOutput[i]))
                    # calculate difference
                    diff = teachingOutput[i]-guess
                    # calculate gradient
                    grad = speed*diff*activation

                    #print("Zum Input: "+str(teachingInput[i])+" ist der Unterschied: "+str(diff)+" grad: "+str(grad))
                    # update the respective weight:
                    self.weightsWithGradient[neuron] = self.weightsWithGradient[neuron]+grad

                #tell outputlayer # online learning (after EACH and EVERY set)
                self.outputLayer.introduceWeightMatrix(self.weightsWithGradient)

            #tell outputlayer # offline learning (after a whole epoch)

            if saveAfterEachEpoch:
                plotWithGrad(self,origInput,origOutput,True, counter)


            # increment counter
            counter = counter + 1

            #randomize trainingsdata
            teachingInput, teachingOutput = randomizeTrainingData(teachingInput,teachingOutput)
            print("\t Finished epoch "+str(counter)+" von "+str(epoch) + " with speed "+str(speed))

            # adjust speed after each epoch
            speed = 0.9 * speed

        t2 = time.time()
        print("Gradient descent took: "+str(t2-t1)+" s")


def randomizeTrainingData(inputs, outputs):
    length = len(inputs)
    for x in range(int(2*length)):
        new1 = np.random.randint(0,length)
        new2 = np.random.randint(0,length)
        inputs[new1], inputs[new2] = inputs[new2],inputs[new1]
        outputs[new1], outputs[new2] = outputs[new2],outputs[new1]
    return inputs,outputs



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
                    np.square(2*np.sin(values) -3*np.cos(3*values) +np.exp(-values*values) -2.3*np.sin(3*values)-np.exp(-(values-4)*(values-4))+np.exp(-3*(values-1))+np.log(values))-10
                    #np.sqrt(values)
                    )
    return y


def saveConfig(model, distance, width):
    """
    save a model
    """
    neurons = np.array([distance,width])
    np.savez("rbf.npz",name1 = model.outputLayer.weightMatrix, name2 = model.RBFLayer.centers, name3 = neurons)
    print("config saved")

def loadModel():
    """
    load a saved model
    """
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
    """
    Plot how the model models the function just with the initial weights
    """
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
    """
    Plot the desired function and how the model models it
    if savePlot is true the picture is saved instead of shown
    """
    maxX = np.amax(x)
    correct = np.array([])
    newGuessed = np.array([])
    usedRange = np.arange(0.01,maxX,0.01)
    for testInput in usedRange:
        newGuessed = np.append(newGuessed, myRbf.guessOutput(testInput))
        correct = np.append(correct, wishFunction(testInput))
    
    # show input neurons
    myRbf.RBFLayer.plotNeurons()
    
    #plt.plot(x,oldGuess, color = "r")
    plt.plot(usedRange,correct, color = "g")
    plt.plot(usedRange,newGuessed, color = "b")   

    if savePlot == True:
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "Plots/Save/Epoch "+str(epochs)+".png"
        abs_file_path = os.path.join(script_dir, rel_path)      
        plt.savefig(abs_file_path)
        #print("\t Plot saved")
    else:
        plt.show()



def performeGrad(rbf,numEpochs, usedRange, saveAfterEachEpoch):
    """
    Do the gradient descent algorithm on the model
    """
    # performe grad
    rbf.gradientDescent(usedRange, wishFunction(usedRange), numEpochs, saveAfterEachEpoch)
   

def getModel(numInputNeurons,centers,width,numOutputNeurons):
    """
    returns rbf model with 1 neuron in in and output layer
    """
     # define single layers
    myInputLayer = inputLayer(1)
    myrbfLayer = RBFLayer(centers, width)
    myOutputLayer = outputLayer(1)

    # get model
    return(RBF(myInputLayer,myrbfLayer,myOutputLayer))

def measureAcc(model, inputs, desiredOutputs, margin):
    """
    Measure how well the model suits the wish function
    """
    numInputs = len(inputs)
    #print("Number of inputs:")
    #print(numInputs)
    outOfMargin = 0
    counter = 0
    for input in inputs:
        guess = model.guessOutput(input)
        distance = abs(desiredOutputs[counter]-guess)
        if distance>margin:#guess*margin:
            #print("passt nicht: ")
            #print(guess)
            #print(distance)
            #print(desiredOutputs[counter])
            outOfMargin = outOfMargin + 1
            
        counter = counter + 1

    print(outOfMargin)
    return (100*(1-outOfMargin/numInputs))



def startFromScrap(xMax,epochs,distanceBetweenInputs,gradSteps,distanceBetweenCenters,savePlot):
    # Determine with how many values you want to estimate first set of weights
    xmax = xMax
    inputs = np.arange(0.05,xmax,distanceBetweenInputs)
    trainings_outputs = wishFunction(inputs)

    # Distribute centers evenly
    distanceBetween = distanceBetweenCenters
    centers = np.arange(0,xmax,distanceBetween)
    width = distanceBetween*0.66 # as recommended in the paper 66%
    # get Model
    myRbf = getModel(1,centers,width,1)

    

    # get weights
    myRbf.calculateWeights(inputs,trainings_outputs)

    # plot initial fit with only the estimated weights
    plotRange =np.arange(0.05,xmax,0.005)
    oldGuess = plotWithoutGrad(myRbf,plotRange)

    # Performe grad descent
    numEpochs = epochs
    gradRange = np.arange(0.05,xmax,gradSteps)
    performeGrad(myRbf, numEpochs, gradRange,savePlot)

    # save calculated weights
    #saveConfig(myRbf, distanceBetween,width)

     # plot with new AND old weights
    plotWithGrad(myRbf,plotRange,oldGuess,savePlot,numEpochs)
    margin = 2 # in absolute
    print("The accuracity with margin "+str(margin)+ " is : " +str(round(measureAcc(myRbf,gradRange,wishFunction(gradRange),margin),2))+"%")





if __name__ == "__main__":
    # get plots with respect to numEpochs
    # startFromScrap(xMax, epochs,distanceBetweenInputs, gradSteps, distanceBetweenCenters, savePlotAfterEachIteration)
    startFromScrap(15, 10, 1, 0.25, 0.4, True)
    
    #plt.plot(wishFunction(np.arange(0.5,15,0.01)))
    #plt.show()

    # load weights and centers
    #weights,centers = loadWeights()
    #myRbf.outputLayer.introduceWeightMatrix(weights)

   
   


