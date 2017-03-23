from network.NeuralLayer import NeuralLayer
from network.CustomLayers import FullyConnectedLayer
from network.CustomLayers import InputLayer
from network.CustomLayers import OutputLayer
from network.Matrix import Matrix
from network import constants

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.inputLayer = InputLayer(None, 1)
        self.finalFCL = FullyConnectedLayer(neuronCount = 1)
        self.outputLayer = OutputLayer([4, 3])
    
    def run(self, input: Matrix):
        
        data = self.inputLayer.process(input)
        for layer in self.layers:
            data = layer.process(data)

        data = self.finalFCL.process(data)
        votes = self.outputLayer.process(data)

        # Return your decision based on the vote
        return 1 if votes[0] > votes[1] else 0

    def train(self, input: Matrix, expectedOutput):        

        error = 0
        running = True
        while running:           
            data = self.inputLayer.process(input)
            for layer in self.layers:
                layer.setError(error)

                # import pdb;pdb.set_trace()
                data = layer.process(data)

            data = self.finalFCL.process(data)

            votes = self.outputLayer.process(data)
            error = abs(expectedOutput[0] - votes[0]) + abs(expectedOutput[1] - votes[1])
            running = error > constants.EPSILON

            print ("Output dim: " + str(data[0].size()) + " ;Error " + str(error))
            # import pdb;pdb.set_trace()

        print ("Final Votes: " + str(votes))
        print("\n\nHooray!!! we're done! Final Error: " + str(error));

    def inputFunction(input):
        """
        Simply just passes the inputs through
        """
        return input