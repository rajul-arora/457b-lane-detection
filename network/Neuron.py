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
