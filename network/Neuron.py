class Neuron:

    def __init__(self, func):
        self.func = func
        self.previousInput = None
        self.error = 0

    def setError(self, error):
        self.error = error

    def process(self, input):
        self.adjustWeights(self.previousInput, self.error)
        self.previousInput = input
        return self.callFunc(input)

    def adjustWeights(self, prevInput, error):
        pass

    def callFunc(self, input):
        return self.func(input)