from network.NeuralLayer import NeuralLayer
from network.FullyConnectedLayer import FullyConnectedLayer
from network.InputLayer import InputLayer
from network.OutputLayer import OutputLayer
from network.Matrix import Matrix
from network import constants

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.inputLayer = InputLayer(None, neuronCount = 1)
        self.finalFCL = FullyConnectedLayer(activation = constants.sigmoid, neuronCount = 2)
        # self.outputLayer = OutputLayer([4, 3])
    
    def run(self, input: Matrix):
        data = self.inputLayer.process(input)
        for layer in self.layers:
            data = layer.process(data)

        data = self.finalFCL.process(data)
        outputs = self.outputLayer.process(data)

        # Return your decision based on the vote
        return 1 if outputs[0] > outputs[1] else 0

    def train(self, input: Matrix, expectedOutput):        

        error = 0
        running = True

        while running:           
            data = self.inputLayer.process(input)

            for layer in self.layers:
                data = layer.process(data)

            output = self.finalFCL.process(data)
            # outputs = self.outputLayer.process(data)
            error = self.calculateLoss(output, expectedOutput)
            running = error > constants.EPSILON

            if running: 
                import pdb;pdb.set_trace()
                deltas = self.finalFCL.calculateDeltas(output, expectedOutput)
                for layer in reversed(self.layers):
                    layer.adjustWeights(deltas)
                    deltas = layer.calculateDeltas(deltas)

            print ("Output dim: " + str(data[0].size()) + " Error " + str(error))
            # import pdb;pdb.set_trace()

        print ("Final Outputs: " + str(output))
        print("\n\nHooray!!! we're done! Final Error: " + str(error));

    
    def calculateLoss(self, output, expectedOutput):
        return self.mse(output, expectedOutput)

    @staticmethod
    def mse(output, expectedOutput):
        """
        Mean Squared Error
        """
        assert len(output) == len(expectedOutput)

        result = 0
        for i in range(len(output)):
            result += (expectedOutput[i] - output[i]) ** 2

        return result/2

    def sum(output, expectedOutput):        
        assert len(output) == len(expectedOutput)

        result = 0
        for i in range(len(output)):
            result += abs(expectedOutput[i] - output[i])

        return result
        

    def inputFunction(input):
        """
        Simply just passes the inputs through
        """
        return input