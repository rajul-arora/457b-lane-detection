from network.Matrix import Matrix
from network import constants
import random

class Neuron:

    def __init__(self, func, activation = None, numForwardNeurons=1):
        self.func = func
        self.activation = activation
        self.previousInput = None
        self.numForwardNeurons = numForwardNeurons
        self.prevResult = [0 for i in range(numForwardNeurons)]
        self.currentValue = None

    def process(self, input):
        self.previousInput = input

        value = self.callFunc(input)
        if self.activation is not None:
            value = self.activation(value)

        self.currentValue = value
        return value


        # result = [0 for i in range(self.numForwardNeurons)]
        # for i in range(self.numForwardNeurons):
        #     result[i] = self.callFunc(input)
        #     if self.activation is not None:
        #         result = self.activation(result)
        #
        #     self.prevResult[i] = result[i]
        #
        # return result

    def adjustWeights(self, deltas):
        pass

    def callFunc(self, input):
        return self.func(input)
