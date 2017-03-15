import math
from network.Neuron import Neuron
from network.NeuralLayer import NeuralLayer
from network.Network import Network

EPSILON = 0.0001

def convolution(input, feature):
    
    numPixelsInFeature = len(feature) * len(feature[0])
    numColMovements = len(input) - len(feature)
    numRowMovements = len(input[0]) - len(feature[0])

    output = []
    # Loop over the outer input marix
    for i in range(numRowMovements):
        for j in range(numColMovements):
            sum = 0
            # Loop over the freature matrix
            for x in range(len(feature)):
                for y in range(len(feature[0])):
                    sum += input[y + j][x + i] * feature[x][y]
            
            result = sum / numPixelsInFeature


    sum = 0
    numPixels = len(inputs)
    for i in range(numPixels):
        sum += inputs[i] * feature[i]

    return sum / numPixels

def sigmoid(input):
    """
    Performs sigmoid on all elements in input and returns a matrix of the same size
    """

    # Instantiate output as a matrix same dimensions as input
    output = [ [0 for i in range(len(input))] for j in range(len(inputs[0])) ] 

    # Perform sigmoid on all elements in input matrix
    for x in range(len(input)):
        for y in range(len(input[0])):
            output[x][y] = 1 /  (1 + math.exp(-1 * input[x][y])) 

    return output

def pool():
    pass

def emptyFunction(inputs):
    return inputs

def main():

    numbeOfPixels = 32

    inputLayer = NeuralLayer(numbeOfPixels, emptyFunction)
    convLayer = NeuralLayer(1, convolution)
    activLayer = NeuralLayer(1, sigmoid)
    poolLayer = NeuralLayer(1, pool)
    fullyConnectedLayer = NeuralLayer(1, emptyFunction)

    layers = [inputLayer, convLayer, activLayer, poolLayer, fullyConnectedLayer]
    network = Network(16, layers)

if __name__ == '__main__':
    main()