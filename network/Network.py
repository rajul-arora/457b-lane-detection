from network.NeuralLayer import NeuralLayer

class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def train(self, input):        

        inputLayer = NeuralLayer(numbeOfPixels, self.inputFunction)
        outputLayer = NeuralLayer(1, self.outputFunction)

        data = inputLayer.process(input)
        for layer in layers:
            data = layer.process(data)

        

    def inputFunction(input):
        """
        Simply just passes the inputs through
        """
        return input

    def outputFunction(input):
        pass