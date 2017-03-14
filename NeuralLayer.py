class NeuralLayer:
    
    def __init__(neuronCount: int, func: Callable):
        self.func = func
        self.neurons = []
        for i in range(neuronCount):
            neurons.append(Neuron(func))
            
    def process(inputs) -> []:
        
        outputs = []
        for neuron in self.neurons:
            result = neuron.process(inputs, weights)
            outputs.append(result)
        
        return outputs