from network.Matrix import Matrix
from network import constants
import random

class Neuron:

    def __init__(self, func, activation = None):
        self.func = func
        self.activation = activation
        self.previousInput = None
        self.prevResult = 0

    def process(self, input):
        self.previousInput = input
        result = self.callFunc(input)
        if self.activation != None:
            result = self.activation(result)
        
        self.prevResult = result
        return result

    def adjustWeights(self, deltas):
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

    def adjustWeights(self, deltas):
        """
        Adjusts the weights using the values of delta propagated to it.
        """
        # import pdb;pdb.set_trace()
        assert len(deltas) == (self.weights.height() * self.weights.width())

        for y in range(0, self.weights.height()):
            for x in range(0, self.weights.width()):
                index = x + (y * self.weights.width()) 
                self.weights[y][x] += constants.LEARNING_RATE * deltas[index] * self.prevResult