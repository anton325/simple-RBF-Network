class outputLayer:
    def __init__(self,numberNeurons):
        self.numberNeurons = numberNeurons
        
    
    def introduceWeightMatrix(self, weightMatrix):
        self.weightMatrix = weightMatrix

    

    def calculateOutput(self,resultFromRBFLayer):
        for i in range(np.size(self.weightMatrix)):
            print("Hi")            
    