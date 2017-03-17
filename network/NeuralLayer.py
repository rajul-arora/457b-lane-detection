from network.Neuron import Neuron

class NeuralLayer:
    
    def __init__(self, nueronCount, func):
        self.neurons = [ self.createNueron(func) for i in range(0, nueronCount) ]
    
    def createNueron(self, func):
        return Neuron(func)

    def process(inputs) -> []:
        assert len(inputs) == len(self.neurons)

        outputs = []
        for i in range(len(inputs)):
            neuron = self.neurons[i]

            # Process each neuron with the corresponding input
            result = neuron.process(inputs[i])
            outputs.append(result)
        
        return outputs