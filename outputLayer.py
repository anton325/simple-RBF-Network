import numpy as np

class outputLayer:
    def __init__(self,numberNeurons):
        self.numberNeurons = numberNeurons
        
    
    def introduceWeightMatrix(self, weightMatrix):
        self.weightMatrix = weightMatrix

    

    def calculateOutput(self,resultFromRBFLayer):
        result = 0
        for i in range(np.size(self.weightMatrix)):
            result = result + resultFromRBFLayer[i]*self.weightMatrix[i]
            print("resultat: "+str(result))
        return result
    