import numpy as np
import matplotlib.pyplot as plt

class RBFLayer:
    def __init__(self, centers, widths):
        """
        save centers and widths
        save number of neurons
        """
        self.centers = centers
        self.widths = widths
        print("centers: "+str(self.centers))
        print("width: "+str(self.widths))
        self.numberOfNeurons = np.size(self.centers)

    
    def outputMatrix(self,input):
        """
        get output matrix
        columns: the neurons
        rows: response of the neurons to a given input
        """
        out = np.array([])
        print("")
        for i in range(np.size(input)):
            # for each input
            for j in range(np.size(self.centers)):
                # for each center
                dis = self.distance(input[i],self.centers[j])
                gaus = self.gaussian(dis)
                
                # ignore very small values
                if gaus<0.001:
                    gaus = 0
                out = np.append(out,gaus)
                print("")
            print("")

        # reshape output
        # reshape: number of columns = anzahl neuronen
        #          number of rows= anzahl inputs
        out = np.reshape(out,(np.size(input),np.size(self.centers)))
        print("Activation matrix of rbf layer: \n"+str(out))

        return out


  
    def outputToScalar(self,scalar):
        """
        Get missing output to arbitrary scalar
        """
        singleValues = np.array([])
        for j in range(np.size(self.centers)):
            neuronOutput = self.getOutputOfSingleNeuron(self.centers[j], scalar)
            singleValues = np.append(singleValues, neuronOutput)
        return singleValues

    
    def getOutputOfSingleNeuron(self,center,scalar):
        """
        get output of single neuron
        @center: which neuron
        """
        dis = self.distance(scalar,center)
        gaus = self.gaussian(dis)
        return gaus


    
    def distance(self,x,center):
        """
        calculate distance between input and neuron
        """
        dis = np.abs(x-center)
        return dis

    
    def gaussian(self,rh):
        """
        calculate output of neuron
        """
        exponent = -np.square(rh)/(2*np.square(self.widths))
        #print("exponent: "+str(exponent))
        res = np.exp(exponent)
        #print("gaus: "+str(res))
        return res

    
    
    def plotNeurons(self):
        """
        draw the neurons in the plot (1D)
        """    
        ax = plt.gca()
        ax.cla()
        for i in range(np.size(self.centers)):
            ax.add_artist(plt.Circle((self.centers[i],0),self.widths, color = "b"))
        
        #print("Plot circles")
        ax.plot()



if __name__ == "__main__":
    centers = np.arange(0.5,4.6,1)
    inputs = np.arange(0,6,1)
    trainings_outputs = np.sin(inputs)

    widths = 0.7 
    for x,y in enumerate(inputs,1):
        # x wird 1, y wird inputs
        print(str(x)+" "+str(y))
    rbfL = RBFLayer(centers,widths)

    print(rbfL.outputMatrix(inputs))
    print(centers)
    print("end")
    