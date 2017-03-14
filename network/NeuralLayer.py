from network.Neuron import Neuron

class NeuralLayer:
    
    def __init__(self, neuronCount, func):
        self.func = func
        self.neurons = [ Neuron(func) for i in range(0,neuronCount) ]
            
    def process(inputs) -> []:
        
        outputs = []
        for neuron in self.neurons:
            result = neuron.process(inputs, weights)
            outputs.append(result)
        
        return outputs