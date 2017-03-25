from network.Matrix import Matrix
from network import constants
import random

class Neuron:

    def __init__(self, func, activation = None):
        self.func = func
        self.activation = activation
        self.previousInput = None
        self.error = 0

    def setError(self, error):
        self.error = error

    def process(self, input):
        self.adjustWeights(self.previousInput, self.error)
        self.previousInput = input
        result = self.callFunc(input)
        if self.activation != None:
            result = self.activation(result)
        
        return result

    def adjustWeights(self, prevInput, error):
        pass

    def callFunc(self, input):
        return self.func(input)

class WeightedNeuron(Neuron):
    def __init__(self, func, activation = None, weights = None, weightsDim = [16, 16]):
        super(WeightedNeuron, self).__init__(func, activation)
        self.weights = weights if (weights != None) else self.generateWeights(weightsDim)

    def callFunc(self, input):
        return self.func(input, self.weights)

    def generateWeights(self, weightsDim):
        width = weightsDim[0]
        height = weightsDim[1]

        # output = [[0 for x in range(width)] for y in range(height)]
        output = Matrix(weightsDim)
        for i in range(height):
            for j in range(width):
                output[i][j] = random.uniform(0, 1)

        return output

    def adjustWeights(self, prevInput, error):
        self.weights = constants.adeline(prevInput, error, self.weights)        
