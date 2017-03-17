from network.Neuron import Neuron
from network.NeuralLayer import NeuralLayer
import random

class WeightedNeuron(Neuron):
    def __init__(self, func, weights = None, weightsDim = [3, 3]):
        super(WeightedNeuron, self).__init__(func)
        self.weights = weights if (weights != None) else self.generateWeights(weightsDim)

    def callFunc(self, input):
        self.func(input, self.weights)

    def generateWeights(self, weightsDim):
        width = weightsDim[0]
        height = weightsDim[1]

        output = [[0 for x in range(width)] for y in range(height)]
        for i in range(width):
            for j in range(height):
                output[i][j] = random.uniform(0, 1)

        return output

    def adjustWeights(self, prevInput, error):
        self.weights = adeline(prevInput, error, weights)        

class ConvolutionLayer(NeuralLayer):
    
    def __init__(self, neuronCount):
        super(ConvolutionLayer, self).__init__(neuronCount, self.convolve)
            
    def createNueron(self, func):
        return WeightedNeuron(func)

    @staticmethod
    def convolve(input, feature):
        """
        Convolves the input matrix with the given feature matrix.
        Returns the convolution as an output matrix (smaller in size from input)
        """
        output = [[0 for x in range(len(input) - len(feature) + 1)] for y in range(len(input[0]) - len(feature[0]) + 1)]
        denom = len(feature) * len(feature[0])

        for i in range(len(input) - len(feature) + 1):
            for j in range(len(input[0]) - len(feature[0]) + 1):
                sum = 0
                for x in range(len(feature)):
                    for y in range(len(feature[0])):
                        sum += input[x+j][y+j] * feature[x][y]

                output[i][j] = sum / denom

        return output

class FullyConnectedLayer(NeuralLayer):
    
    def __init__(self, neuronCount):
        super(FullyConnectedLayer, self).__init__(neuronCount, self.combine)

    def process(self, inputs):
        """
        Passes the inputs to all nuerons.
        """
        outputs = []
        for neuron in self.neurons:
            # Process each neuron with all the inputs
            result = neuron.process(inputs)
            outputs.append(result)
        
        return outputs

    @staticmethod
    def combine(inputs):
        """
        Takes a list of 2d arrays and smushes their entries together into a 2d array
        Each row is each input as a a 1d array
        """
        output = []
        for input in inputs:

            vector = []
            for i in range(len(input)):
                for j in range(len(input[i])):
                    vector.append(input[i][j])

            output.append(vector)

        return output

class OutputLayer(FullyConnectedLayer):
    
    def __init__(self, neuronCount):
        super(OutputLayer, self).__init__(neuronCount, self.vote)
            
    def createNueron(self, func):
        return WeightedNeuron(func)

    @staticmethod
    def vote(input, weights):
        """
        Uses the input matrix and weights to vote
        """
        
        denom = len(weights) * len(weights[0])
        sum = 0
        for x in range(len(weights)):
            for y in range(len(weights[0])):
                sum += input[x][y] * weights[x][y]

        return sum / denom
