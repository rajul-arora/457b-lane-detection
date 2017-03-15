from network.Neuron import Neuron

class NeuralLayer:
    
    def __init__(self, neuronCount, func, hasWeights = False):
        self.neurons = [ Neuron(func, hasWeights) for i in range(0,neuronCount) ]
            
    def process(inputs) -> []:
        assert len(inputs) == len(self.neurons)

        outputs = []
        for i in range(len(inputs)):
            neuron = self.neurons[i]

            # Process each neuron with the corresponding input
            result = neuron.process(inputs[i])
            outputs.append(result)
        
        return outputs