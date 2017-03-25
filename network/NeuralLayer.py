from network.Neuron import Neuron
from network.Matrix import Matrix
from network import constants

class NeuralLayer:
    
    def __init__(self, func, activation = None, neuronCount = constants.NEURON_COUNT):
        self.neurons = [ self.createNueron(func, activation) for i in range(0, neuronCount) ]
    
    def createNueron(self, func, activation):
        return Neuron(func, activation)

    def setError(self, error):
        for neuron in self.neurons:
            neuron.setError(error)

    def process(self, inputs):
        """
        Passes the inputs to their corresponding neuron.
        That is, input[i] -> neuron[i]
        """
        assert len(inputs) == len(self.neurons)

        outputs = []
        for i in range(len(inputs)):
            neuron = self.neurons[i]

            # Process each neuron with the corresponding input
            result = neuron.process(inputs[i])
            outputs.append(result)
        
        
        # print("Input to layer(" + str(self) + "): " + str(inputs))
        # print("Output from layer(" + str(self) + "): " + str(outputs))
        return outputs