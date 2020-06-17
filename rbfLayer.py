import numpy as np

class RBFLayer:
    def __init__(self, centers, widths):
        self.centers = centers
        self.widths = widths
        print("centers: "+str(self.centers))
        print("width: "+str(self.widths))
        self.numberOfNeurons = np.size(self.centers)

    def outputMatrix(self,input):
        out = np.array([])
        print("")
        for i in range(np.size(input)):
            for j in range(np.size(self.centers)):
                # first do it all for input1, than for input2...
                dis = self.distance(input[i],self.centers[j])
                #print("dis: "+str(dis))
                gaus = self.gaussian(dis)
                #print("gaus: "+str(gaus))
                if gaus<0.001:
                    gaus = 0
                out = np.append(out,gaus)
                print("")
            print("")
        print("out direkt nach schleifen: \n"+str(out))
        
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
    """
    def getOutputOfSingleNeuron(self,center,scalar):
        dis = self.distance(scalar,center)
        gaus = self.gaussian(dis)
        return gaus


            

    def distance(self,x,center):
        dis = np.abs(x-center)
        #print("center: "+str(center)+" x: "+str(x))
        """for xValue,c in x,center:
            dis = dis + np.square(xValue-c)
        dis = np.sqrt(dis)"""
        return dis


    def gaussian(self,rh):
        exponent = -np.square(rh)/(2*np.square(self.widths))
        #print("exponent: "+str(exponent))
        res = np.exp(exponent)
        #print("gaus: "+str(res))
        return res

    










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
    