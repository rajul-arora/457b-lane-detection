from network.NeuralLayer import NeuralLayer
from network.CustomLayers import FullyConnectedLayer
from network.CustomLayers import OutputLayer

class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def train(self, input, expectedOutput):        

        inputLayer = NeuralLayer(numbeOfPixels, self.inputFunction)
        finalFCL = FullyConnectedLayer(1)
        outputLayer = OutputLayer(2)

        data = inputLayer.process(input)
        for layer in layers:
            data = layer.process(data)

        data = finalFCL.process(data)
        votes = outputLayer.process(data)
        error = (expectedOutput[0] - votes[0]) + (expectedOutput[1] - votes[1])

    def inputFunction(input):
        """
        Simply just passes the inputs through
        """
        return input