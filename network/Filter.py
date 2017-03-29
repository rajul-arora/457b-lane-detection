import math
from network.Matrix import Matrix
from network.WeightedNeuron import WeightedNeuron
from network import constants

class Filter:

    def __init__(self, activation = None, numNeurons = 9):
        self.activation = activation
        width = int(math.sqrt(numNeurons))
        self.neurons = Matrix([width, width])
        self.previousInput = None
        self.prevResult = None

        for i in range(0, width):
            for j in range(0, width):
                self.neurons[i][j] = WeightedNeuron(constants.multiply, activation, weightsDim=[1,1])

    def adjustNeuronWeights(self, deltas):
        assert len(deltas) == self.neurons.width() * self.neurons.height()

        for y in range(0, self.neurons.height()):
            for x in range(0, self.neurons.width()):
                index = x + (y * self.neurons.width())
                self.neurons[y][x].adjustWeights(deltas[index])

    def process(self, input):
        self.previousInput = input
        result = Filter.convolve(input, self.neurons)
        if self.activation != None:
            result = self.activation(result)

        self.prevResult = result
        return result

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
                        sum += feature[x][y].multiply(input[x+i][y+j])[0][0]

                output[i][j] = sum / denom
        return output
