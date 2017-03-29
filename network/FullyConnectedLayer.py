from network.WeightedNeuron import WeightedNeuron
from network.NeuralLayer import NeuralLayer
from network.Matrix import Matrix
from network import constants
import math

class FullyConnectedLayer(NeuralLayer):
    
    def __init__(self, activation = None, neuronCount = constants.NEURON_COUNT, numForwardNeurons=1):
        super(FullyConnectedLayer, self).__init__(func = self.combine, activation = activation, neuronCount = neuronCount, numForwardNeurons=numForwardNeurons)

    def createNueron(self, func, activation, numForwardNeurons):
        return WeightedNeuron(func, activation, numForwardNeurons, weightsDim=[32,32])

    def process(self, inputs):
        """
        Passes the inputs to all neurons.
        """
        outputs = []
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            # Process each neuron with all the inputs
            neuronInputs = []
            for input in inputs:
                neuronInputs.append(input[i])
            result = neuron.process(neuronInputs)

            # Take the dot product of result with weights to get output
            for j in range(0, self.numForwardNeurons):
                neuron.prevResults[i] = self.dotProduct(result, neuron.weights[j])

            outputs.append(neuron.prevResults)
        
        return outputs

    def calculateDeltas(self, prevDeltas):
        """
        Calculates the delta for the given output and expected output.
        """
        assert len(prevDeltas) == len(self.neurons)

        deltas = []

        for neuron in self.neurons:
            sum = 0
            for i in range(0,len(deltas)):
                sum += deltas[i] * neuron.weights[i]

            delta = constants.sigderiv(neuron.prevResult) * sum
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
        sum = 0
        flat1 = input.flatten()
        flat2 = weights.flatten()

        for x in range(min(len(flat1), len(flat2))):
            sum += flat1[x] * flat2[x]

        return sum