from network.NeuralLayer import NeuralLayer
from network import constants

class InputLayer(NeuralLayer):
    
    def process(self, input):
        
        outputs = []
        for i in range(constants.NEURON_COUNT):
            outputs.append(input)
        
        return outputs