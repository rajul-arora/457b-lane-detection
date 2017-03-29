from network.WeightedNeuron import WeightedNeuron
from network.NeuralLayer import NeuralLayer
from network.Matrix import Matrix
from network import constants
import math

class FullyConnectedLayer(NeuralLayer):
    
    def __init__(self, activation = None, neuronCount = constants.NEURON_COUNT):
        super(FullyConnectedLayer, self).__init__(func = self.combine, activation = activation, neuronCount = neuronCount)

    def createNueron(self, func, activation):
        return WeightedNeuron(func, activation)

    def process(self, inputs):
        """
        Passes the inputs to all nuerons.
        """
        outputs = []
        for neuron in self.neurons:
            # Process each neuron with all the inputs
            result = neuron.process(inputs)

            # Take the dot product of result with weights to get output
            result = self.dotProduct(result, neuron.weights)
            neuron.prevResult = result
            outputs.append(result)
        
        return outputs

    def calculateDeltas(self, output, expectedOutput): 
        """
        Calculates the delta for the given output and expected output.
        """
        assert(len(output) == len(expectedOutput))
        deltas = []
        # import pdb;pdb.set_trace()
        for i in range(0, len(output)):
            delta = constants.sigderiv(self.neurons[i].prevResult) * (expectedOutput[i] - output[i])
            deltas.append(delta)
        return deltas

    @staticmethod
    def combine(inputs, empty):
        """
        Takes a list of matrices and smushes their entries together into a single matrix
        Each row is each input as a a 1d array
        """
        output = []
        for input in inputs:
            vector = []
            for i in range(input.height()):
                for j in range(input.width()):
                    vector.append(input[i][j])
                    
            output.append(vector)
        
        return Matrix.convert(output)

    @staticmethod
    def dotProduct(input: Matrix, weights: Matrix):
        """
        Performs dot-product of input with weights
        """
        weightsDim = weights.size()
        denom = weightsDim[0] * weightsDim[1]
        sum = 0
        for x in range(min(input.height(), weights.height())):
            for y in range(min(input.width(), weights.width())):
                sum += input[x][y] * weights[x][y]

        return sum / denom