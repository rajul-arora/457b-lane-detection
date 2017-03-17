class Neuron:

    def __init__(self, func):
        self.func = func

    def process(self, inputs):
        self.callFunc(inputs)

    def callFunc(self, inputs):
        self.func(inputs)