class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def train(input):
        
        inputStep = input
        for layer in layers:
            inputStep = layer.process(inputStep)
