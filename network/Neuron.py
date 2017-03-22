from network.Matrix import Matrix
from network import constants
import random

class Neuron:

    def __init__(self, func):
        self.func = func
        self.previousInput = None
        self.error = 0

    def setError(self, error):
        self.error = error

    def process(self, input):
        self.adjustWeights(self.previousInput, self.error)
        self.previousInput = input
        return self.callFunc(input)

    def adjustWeights(self, prevInput, error):
        pass

    def callFunc(self, input):
        return self.func(input)

class WeightedNeuron(Neuron):
    def __init__(self, func, weights = None, weightsDim = [3, 3]):
        super(WeightedNeuron, self).__init__(func)
        self.weights = weights if (weights != None) else self.generateWeights(weightsDim)

    def callFunc(self, input):
        return self.func(input, self.weights)

    def generateWeights(self, weightsDim):
        width = weightsDim[0]
        height = weightsDim[1]

        # output = [[0 for x in range(width)] for y in range(height)]
        output = Matrix(weightsDim)
        for i in range(width):
            for j in range(height):
                output[i][j] = random.uniform(0, 1)

        return output

    def adjustWeights(self, prevInput, error):
        self.weights = constants.adeline(prevInput, error, self.weights)        
