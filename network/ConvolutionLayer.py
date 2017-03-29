from network.Neuron import Neuron
from network.Neuron import WeightedNeuron
from network.NeuralLayer import NeuralLayer
from network.Matrix import Matrix
from network import constants
import math

class ConvolutionLayer(NeuralLayer):
    
    def __init__(self, activation, neuronsPerFilter = 9):
        size = math.ceil(math.sqrt(neuronsPerFilter))
        self.weightsDim = [size, size]
        super(ConvolutionLayer, self).__init__(self.convolve, activation)
            
    def createNueron(self, func, activation):
        return WeightedNeuron(func = func, activation = activation, weightsDim = self.weightsDim)

    def calculateDeltas(self, prevDeltas): 
        """
        Calculates the delta for the given output and expected output.
        """
        assert len(prevDeltas) == len(self.nuerons)
        
        deltas = []
        for i in range(0, len(prevDeltas)):
            delta = constants.sigderiv(self.neurons[i].prevResult) * prevDeltas[i]
            deltas.append(delta)
        return deltas 

    @staticmethod
    def convolve(input: Matrix, feature: Matrix):
        """
        Convolves the input matrix with the given feature matrix.
        Returns the convolution as an output matrix (smaller in size from input)
        """
        outputWidth = input.width() - feature.width() + 1
        outputHeight = input.height() - feature.height() + 1

        # output = [[0 for x in range(len(input) - len(feature) + 1)] for y in range(len(input[0]) - len(feature[0]) + 1)]
        output = Matrix([outputWidth, outputHeight])
        # denom = len(feature) * len(feature[0])
        featureDim = feature.size()
        denom = featureDim[0] * featureDim[1]

        for i in range(outputHeight):
            for j in range(outputWidth):
                sum = 0
                for x in range(feature.height()):
                    for y in range(feature.width()):
                        sum += input[x+i][y+j] * feature[x][y]

                output[i][j] = sum / denom
        return output