import numpy as np
import matplotlib.pyplot as plt


class RBFLayer:
    """
    save centers and widths
    save number of neurons
    """
    def __init__(self, centers, widths):
        self.centers = centers
        self.widths = widths
        print("centers: "+str(self.centers))
        print("width: "+str(self.widths))
        self.numberOfNeurons = np.size(self.centers)

    """
    Calculate output matrix to given input 
    columns: the neurons
    rows: response of the neurons to a given input
    """
    def outputMatrix(self,input):
        out = np.array([])
        print("")
        # for each input
        for i in range(np.size(input)):
            # for each rbf neuron
            for j in range(np.size(self.centers)):
                # get distance                
                dis = self.distance(input[i],self.centers[j])
                # get activation
                gaus = self.gaussian(dis)
                # ignore small values
                if gaus<0.01:
                    gaus = 0
                out = np.append(out,gaus)
                print("")
            print("")
        print("Activation matrix of rbf layer: \n"+str(out))
        
        # reshape output
        # reshape: anzahl spalten = anzahl neuronen
        #          anzahl zeilen = anzahl inputs
        out = np.reshape(out,(np.size(input),np.size(self.centers)))
        #out = out.T
        return out


    """
    Get missing output to arbitrary scalar
    """
    def outputToScalar(self,scalar):
        #print("scalar: "+str(scalar))
        singleValues = np.array([])
        for j in range(np.size(self.centers)):
            neuronOutput = self.getOutputOfSingleNeuron(self.centers[j], scalar)
            singleValues = np.append(singleValues, neuronOutput)
        return singleValues

    """
    get output of single neuron
    @center: which neuron
    """
    def getOutputOfSingleNeuron(self,center,scalar):
        dis = self.distance(scalar,center)
        gaus = self.gaussian(dis)
        return gaus


    """
    calculate distance between input and neuron
    """
    def distance(self,x,center):
        dis = np.abs(x-center)
        #print("center: "+str(center)+" x: "+str(x))
        """for xValue,c in x,center:
            dis = dis + np.square(xValue-c)
        dis = np.sqrt(dis)"""
        return dis

    """
    calculate output of neuron
    """
    def gaussian(self,rh):
        exponent = -np.square(rh)/(2*np.square(self.widths))
        #print("exponent: "+str(exponent))
        res = np.exp(exponent)
        #print("gaus: "+str(res))
        return res

    """
    draw the neurons on the input (1D)
    """    
    def plotNeurons(self):
        ax = plt.gca()
        ax.cla()
        for i in range(np.size(self.centers)):
            ax.add_artist(plt.Circle((self.centers[i],0),self.widths, color = "y", alpha = 0.5))
        
        print("Plot circles")
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

    print("With new layout: \n"+str(rbfL.outputMatrix(inputs)))
    print("centers: "+str(centers))
    print("end")
    