from network.Neuron import Neuron
from network.WeightedNeuron import WeightedNeuron
from network.Neuron import Neuron
from network.NeuralLayer import NeuralLayer
from network.Filter import Filter
from network.Matrix import Matrix
from network import constants
import math

class ConvolutionLayer:
    
    def __init__(self, activation, numFilters = 3, neuronsPerFilter = 9):
        size = math.ceil(math.sqrt(neuronsPerFilter))
        self.weightsDim = [size, size]
        self.filters = [ Filter(activation, neuronsPerFilter) for i in range(0, numFilters) ]

    def calculateDeltas(self, prevDeltas): 
        """
        Calculates the delta for the given output and expected output.
        """
        pass
        # assert len(prevDeltas) == len(self.filters)
        #
        # deltas = []
        # for i in range(0, len(prevDeltas)):
        #     delta = constants.sigderiv(self.filters[i].prevResult) * prevDeltas[i]
        #     deltas.append(delta)
        # return deltas

    def adjustWeights(self, deltas):
        """"
        Adjusts the weights using the values of delta propagated to it.
        """
        for filter in self.filters:
            filter.adjustNeuronWeights(deltas)

    def process(self, inputs):
        """
        Passes the inputs to their corresponding neuron.
        That is, input[i] -> neuron[i]
        """
        assert len(inputs) == len(self.filters)

        outputs = []
        for i in range(len(inputs)):
            filter = self.filters[i]

            # Process each neuron with the corresponding input
            result = filter.process(inputs[i])
            outputs.append(result)

        # print("Input to layer(" + str(self) + "): " + str(inputs))
        # print("Output from layer(" + str(self) + "): " + str(outputs))
        return outputs

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
