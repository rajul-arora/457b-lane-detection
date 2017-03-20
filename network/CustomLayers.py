from network.Neuron import Neuron
from network.NeuralLayer import NeuralLayer
import random
from network import constants

class WeightedNeuron(Neuron):
    def __init__(self, func, weights = None, weightsDim = [3, 3]):
        super(WeightedNeuron, self).__init__(func)
        self.weights = weights if (weights != None) else self.generateWeights(weightsDim)

    def callFunc(self, input):
        return self.func(input, self.weights)

    def generateWeights(self, weightsDim):
        width = weightsDim[0]
        height = weightsDim[1]

        output = [[0 for x in range(width)] for y in range(height)]
        for i in range(width):
            for j in range(height):
                output[i][j] = random.uniform(0, 1)

        return output

    def adjustWeights(self, prevInput, error):
        self.weights = constants.adeline(prevInput, error, self.weights)        

class ConvolutionLayer(NeuralLayer):
    
    def __init__(self):
        super(ConvolutionLayer, self).__init__(self.convolve)
            
    def createNueron(self, func):
        return WeightedNeuron(func)

    @staticmethod
    def convolve(input, feature):
        """
        Convolves the input matrix with the given feature matrix.
        Returns the convolution as an output matrix (smaller in size from input)
        """
        output = [[0 for x in range(len(input) - len(feature) + 1)] for y in range(len(input[0]) - len(feature[0]) + 1)]
        denom = len(feature) * len(feature[0])

        for i in range(len(input) - len(feature) + 1):
            for j in range(len(input[0]) - len(feature[0]) + 1):
                sum = 0
                for x in range(len(feature)):
                    for y in range(len(feature[0])):
                        sum += input[x+j][y+j] * feature[x][y]

                output[i][j] = sum / denom

        return output

class FullyConnectedLayer(NeuralLayer):
    
    def __init__(self, func = None, neuronCount = constants.NEURON_COUNT):
        super(FullyConnectedLayer, self).__init__(self.combine if (func == None) else func, neuronCount)

    def process(self, inputs):
        """
        Passes the inputs to all nuerons.
        """
        outputs = []
        for neuron in self.neurons:
            # Process each neuron with all the inputs
            result = neuron.process(inputs)
            outputs.append(result)
        
        return outputs

    @staticmethod
    def combine(inputs):
        """
        Takes a list of 2d arrays and smushes their entries together into a 2d array
        Each row is each input as a a 1d array
        """
        # import pdb;pdb.set_trace()
        output = []
        for input in inputs:
            vector = []
            for i in range(len(input)):
                for j in range(len(input[i])):
                    vector.append(input[i][j])
            # pdb.set_trace()
            output.append(vector)
        
        return output

class OutputLayer(NeuralLayer):
    
    def __init__(self):
        super(OutputLayer, self).__init__(self.vote, 2)
            
    def createNueron(self, func):
        return WeightedNeuron(func)

    def process(self, inputs):
        """
        Passes the input to both output voters
        """
        assert len(inputs) == 1

        outputs = []
        for neuron in self.neurons:
            result = neuron.process(inputs[0])
            outputs.append(result)

        return outputs

    @staticmethod
    def vote(input, weights):
        """
        Uses the input matrix and weights to vote
        """
        
        denom = len(weights) * len(weights[0])
        sum = 0
        for x in range(len(weights)):
            for y in range(len(weights[0])):
                # import pdb;pdb.set_trace()
                sum += input[x][y] * weights[x][y]

        return sum / denom

class InputLayer(NeuralLayer):
    
    def process(self, input):
        """
        Calls open cv to get the image
        """
        # Call Open CV here
        image = input
        outputs = []
        for i in range(constants.NEURON_COUNT):
            outputs.append(image)
        
        return outputs