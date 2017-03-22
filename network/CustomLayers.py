from network.Neuron import Neuron
from network.Neuron import WeightedNeuron
from network.NeuralLayer import NeuralLayer
from network.Matrix import Matrix
from network import constants

class ConvolutionLayer(NeuralLayer):
    
    def __init__(self):
        super(ConvolutionLayer, self).__init__(self.convolve)
            
    def createNueron(self, func):
        return WeightedNeuron(func)

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
        Takes a list of matrices and smushes their entries together into a single matrix
        Each row is each input as a a 1d array
        """
        output = []
        for input in inputs:
            vector = []
            for i in range(input.height()):
                for j in range(input.width()):
                    vector.append(input[i][j])
            # pdb.set_trace()
            output.append(vector)
        
        # import pdb; pdb.set_trace();
        return Matrix.convert(output)

class OutputLayer(NeuralLayer):
    
    def __init__(self, votingDim = [3, 3]):
        self.votingDim = votingDim
        super(OutputLayer, self).__init__(self.vote, 2)
            
    def createNueron(self, func):
        return WeightedNeuron(func, weightsDim = self.votingDim)

    def process(self, inputs):
        """
        Passes the input to both output voters
        """
        # import pdb; pdb.set_trace();
        assert len(inputs) == 1

        outputs = []
        for neuron in self.neurons:
            result = neuron.process(inputs[0])
            outputs.append(result)

        return outputs

    @staticmethod
    def vote(input: Matrix, weights: Matrix):
        """
        Uses the input matrix and weights to vote
        """
        weightsDim = weights.size()
        # denom = len(weights) * len(weights[0])
        denom = weightsDim[0] * weightsDim[1]
        sum = 0
        for x in range(min(input.height(), weights.height())):
            for y in range(min(input.width(), weights.width())):
                # import pdb;pdb.set_trace()
                sum += input[x][y] * weights[x][y]

        return sum / denom

class InputLayer(NeuralLayer):
    
    def process(self, input):
        
        outputs = []
        for i in range(constants.NEURON_COUNT):
            outputs.append(input)
        
        return outputs