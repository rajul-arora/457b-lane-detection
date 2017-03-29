from network.Neuron import Neuron
from network import constants
from network.Matrix import Matrix
import random

class WeightedNeuron(Neuron):
    def __init__(self, func, activation = None,  weightsDim = [16, 16], numForwardNeurons = 1):
        super(WeightedNeuron, self).__init__(func, activation, numForwardNeurons = numForwardNeurons)
        self.weights = [0 for i in range(numForwardNeurons)]

        for i in range(numForwardNeurons):
            self.weights[i] = self.generateWeights(weightsDim)

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

    def multiply(self, input):
        multWeights = Matrix(self.weights.size())
        for x in range(0, self.weights.height()):
            for y in range(0, self.weights.width()):
                multWeights[x][y] = self.weights[x][y] * input

        return multWeights

    def adjustWeights(self, deltas):
        """
        Adjusts the weights using the values of delta propagated to it.
        """
        # import pdb;pdb.set_trace()
        assert len(deltas) == len(self.weights)

        for i in range(0, len(self.weights)):
            weight = self.weights[i]
            for y in range(0, weight.height()):
                for x in range(0, weight.width()):
                    weight[y][x] += constants.LEARNING_RATE * deltas[i] * self.prevResult[i]