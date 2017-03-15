class Neuron:

    def __init__(self, func, hasWeights = False):
        self.func = func
        self.weights = hasWeights ? [] : None

    def process(inputs):
        self.func(inputs, self.weights)