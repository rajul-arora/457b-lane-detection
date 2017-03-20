from network.Neuron import Neuron
from network import constants

class NeuralLayer:
    
    def __init__(self, func, neuronCount = constants.NEURON_COUNT):
        self.neurons = [ self.createNueron(func) for i in range(0, neuronCount) ]
    
    def createNueron(self, func):
        return Neuron(func)

    def setError(self, error):
        for neuron in self.neurons:
            neuron.setError(error)

    def process(self, inputs):
        """
        Passes the inputs to their corresponding neuron.
        That is, input[i] -> neuron[i]
        """
        import pdb;pdb.set_trace()
        assert len(inputs) == len(self.neurons)

        outputs = []
        for i in range(len(inputs)):
            neuron = self.neurons[i]

            # Process each neuron with the corresponding input
            result = neuron.process(inputs[i])
            outputs.append(result)
        
        print("Output from layer: " + str(outputs))
        return outputs