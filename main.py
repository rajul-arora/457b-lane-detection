import math
from network.Neuron import Neuron
from network.NeuralLayer import NeuralLayer
from network.Network import Network

EPSILON = 0.0001

def convolution(input, feature):
    output = [[0 for x in range(len(input) - len(feature) + 1)] for y in range(len(input[0]) - len(feature[0]) + 1)]
    denom = len(feature) * len(feature[0])

    for i in range(len(input) - len(feature) + 1):
        for j in range(len(input[0]) - len(feature[0]) + 1):
            sum = 0
            for x in range(len(feature)):
                for y in range(len(feature[0])):
                    sum += input[x+j][y+j] * feature[x][y]

            output[i][j] = sum / denom

    return output

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

    inputMatrix = [
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
    ]

    featureMatrix = [
        [1,0],
        [0,1],
    ]
    
    output = convolution(inputMatrix, featureMatrix)

    for i in range(len(output)):
        print(output[i])

    # numbeOfPixels = 32

    # inputLayer = NeuralLayer(numbeOfPixels, emptyFunction)
    # convLayer = NeuralLayer(1, convolution)
    # activLayer = NeuralLayer(1, sigmoid)
    # poolLayer = NeuralLayer(1, pool)
    # fullyConnectedLayer = NeuralLayer(1, emptyFunction)

    # layers = [inputLayer, convLayer, activLayer, poolLayer, fullyConnectedLayer]
    # network = Network(16, layers)

if __name__ == '__main__':
    main()