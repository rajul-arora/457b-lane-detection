from network.NeuralLayer import NeuralLayer
from network.CustomLayers import FullyConnectedLayer
from network.CustomLayers import InputLayer
from network.CustomLayers import OutputLayer
from network import constants

class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def train(self, input, expectedOutput):        

        numbeOfPixels = len(input) * len(input[0])
        inputLayer = InputLayer(self.inputFunction, 1)
        finalFCL = FullyConnectedLayer(neuronCount = 1)
        outputLayer = OutputLayer()

        data = inputLayer.process(input)
        # import pdb;pdb.set_trace()
        print("Output from input layer: " + str(len(data)))
        error = 0
        running = True
        while running:
            for layer in self.layers:
                layer.setError(error)

                # import pdb;pdb.set_trace()
                data = layer.process(data)
                print("Layer data: " + str(data))

            data = finalFCL.process(data)
            votes = outputLayer.process(data)
            error = (expectedOutput[0] - votes[0]) + (expectedOutput[1] - votes[1])
            running = error > constants.EPSILON
            print ("Error " + str(error))

    def inputFunction(input):
        """
        Simply just passes the inputs through
        """
        return input