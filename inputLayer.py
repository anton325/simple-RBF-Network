import numpy as np
import matplotlib.pyplot as plt

class inputLayer:
    def __init__(self, numberInputNeurons,centers,width):
        self.numberInputNeurons = numberInputNeurons
        self.centers = centers
        self.width = width

    
    def plotNeurons(self):
        ax = plt.gca()
        ax.cla()
        for i in range(np.size(self.centers)):
            ax.add_artist(plt.Circle((self.centers[i],0),self.width, color = "b"))
        print("plot circles")
        ax.plot()

