import numpy as np

class outputLayer:
    def __init__(self,numberNeurons):
        """
        save number of neurons and initialise weight matrix
        """
        self.numberNeurons = numberNeurons
        self.weightMatrix = np.array([])
        
    
    
    def introduceWeightMatrix(self, weightMatrix):
        """
        copy weight matrix
        """
        self.weightMatrix = weightMatrix

    
    
    def calculateOutput(self,resultFromRBFLayer):
        """
        calculate output using the weights and the results from the rbf layer
        """
        result = 0
        for i in range(np.size(self.weightMatrix)):
            result = result + resultFromRBFLayer[i]*self.weightMatrix[i]
            #print("resultat: "+str(result))
        return result
    