import numpy as np

class outputLayer:
    def __init__(self,numberNeurons):
        self.numberNeurons = numberNeurons
        self.weightMatrix = np.array([])
        
    
    """
    copy weight matrix
    """
    def introduceWeightMatrix(self, weightMatrix):
        self.weightMatrix = weightMatrix

    
    """
    calculate output using the weights and the results from the rbf layer
    """
    def calculateOutput(self,resultFromRBFLayer):
        result = 0
        for i in range(np.size(self.weightMatrix)):
            result = result + resultFromRBFLayer[i]*self.weightMatrix[i]
            #print("resultat: "+str(result))
        return result
    