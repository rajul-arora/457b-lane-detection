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

def pool(input):
    """
    Performs pooling on the input matrix and returns a smaller matrix
    """

    windowSize = 2
    stride = 2

    output = []
    for i in range(0, len(input), stride):

        outputRow = []
        for j in range(0, len(input[0]), stride):            
            maxVal = max(input, windowSize, i, j)
            outputRow.append(maxVal)        

        output.append(outputRow)
    
    return output

def max(inputs, windowSize, xOffset, yOffset):

    res = inputs[xOffset][yOffset]
    inputRowLen = len(inputs)
    inputColLen = len(inputs[0])
    for i in range(xOffset, min(windowSize + xOffset, inputRowLen)):
        for j in range(yOffset, min(windowSize + yOffset, inputColLen)):
            if inputs[i][j] > res:
                res = inputs[i][j]
    
    return res

def avg(inputs, windowSize, xOffset, yOffset):
    
    res = 0
    for i in range(xOffset, windowSize + xOffset):
        for j in range(yOffset, windowSize + yOffset):
            res += inputs[i][j]
    
    return res / (windowSize * windowSize)

def emptyFunction(inputs):
    return inputs

def main():

    inputMatrix = [
        [1,0.2,1,0.3,1],
        [0.1,0.4,1,0.5,1],
        [0.2,0.6,0.9,0.7,1],
        [0.3,0.8,0.8,0.9,1],
        [0.4,0.9,0.7,0.99,1],
    ]

    featureMatrix = [
        [1,0],
        [0,1],
    ]
    
    output = pool(inputMatrix)
    convOutput = convolution(inputMatrix, featureMatrix)

    for i in range(len(output)):
        print(output[i])

    for i in range(len(convOutput)):
        print(convOutput[i])


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