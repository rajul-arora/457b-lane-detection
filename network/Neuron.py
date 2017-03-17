class Neuron:

    def __init__(self, func):
        self.func = func

    def setError(self, error):
        self.error = error

    def process(self, input):
        self.adjustWeights(self.previousInput, self.error)
        self.callFunc(input)
        self.previousInput = input

    def adjustWeights(self, prevInput, error):
        pass

    def callFunc(self, input):
        self.func(input)