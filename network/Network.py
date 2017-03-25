from network.NeuralLayer import NeuralLayer
from network.CustomLayers import FullyConnectedLayer
from network.CustomLayers import InputLayer
from network.CustomLayers import OutputLayer
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
                layer.setError(error)

                # import pdb;pdb.set_trace()
                data = layer.process(data)

            outputs = self.finalFCL.process(data)

<<<<<<< e972701c7ba9e3f6c1e715760f021158e7dadbdc
            # outputs = self.outputLayer.process(data)
            error = abs(expectedOutput[0] - outputs[0]) + abs(expectedOutput[1] - outputs[1])
            running = error > constants.EPSILON and False
=======
            votes = self.outputLayer.process(data)
            error = 0.5 * abs(expectedOutput[0] - votes[0]) ** 2 + 0.5 * abs(expectedOutput[1] - votes[1]) ** 2
            running = error > constants.EPSILON
>>>>>>> Created ReLU and Derivative of ReLU functions

            print ("Output dim: " + str(data[0].size()) + " Error " + str(error))
            # import pdb;pdb.set_trace()

        print ("Final Outputs: " + str(outputs))
        print("\n\nHooray!!! we're done! Final Error: " + str(error));

    def calculateLoss(output, expectedOutput):
        return mse(output, expectedOutput)

    def mse(output, expectedOutput):
        """
        Mean Squared Error
        """
        assert len(output) == len(expectedOutput)

        result = 0
        for i in range(len(output)):
            result += (expectedOutput[i] - constants.sigmoid(output[i])) ** 2

        return result/2

    def ReLU(X: Matrix):
        """
        Calculates the ReLU of the given input matrix X.
        """
        output = [[0 for i in range(0,len(X))] for j in range(0,len(X[0]))]

        for i in range(0,len(X)):
            for j in range(0,len(X[0])):
                output[i][j] = max(0, X[i][j])

        return output

    def dReLU(X: Matrix):
        """
        Calculates the derivative of ReLU of the given input matrix X.
        """
        output = [[0 for i in range(0,len(X))] for j in range(0,len(X[0]))]

        for i in range(0,len(X)):
            for j in range(0,len(X[0])):
                output[i][j] = 1 if X[i][j] > 0 else 0

        return output

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