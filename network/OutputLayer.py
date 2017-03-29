from network.Neuron import Neuron
from network.WeightedNeuron import WeightedNeuron
from network.NeuralLayer import NeuralLayer
from network.Matrix import Matrix
from network import constants
import math

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