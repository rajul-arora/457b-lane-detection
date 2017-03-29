from network.Neuron import Neuron
from network.Neuron import WeightedNeuron
from network.NeuralLayer import NeuralLayer
from network.Matrix import Matrix
from network import constants
import math

class InputLayer(NeuralLayer):
    
    def process(self, input):
        
        outputs = []
        for i in range(constants.NEURON_COUNT):
            outputs.append(input)
        
        return outputs