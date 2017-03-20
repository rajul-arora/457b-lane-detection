import math
from network.Neuron import Neuron
from network.NeuralLayer import NeuralLayer
from network.CustomLayers import ConvolutionLayer
from network.Network import Network
import network.constants

def sigmoid(input):
    """
    Performs sigmoid on all elements in input and returns a matrix of the same size
    """

    # Instantiate output as a matrix same dimensions as input
    output = [ [0 for i in range(len(input))] for j in range(len(input[0])) ] 

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

def max(input, windowSize, xOffset, yOffset):

    res = input[xOffset][yOffset]
    inputRowLen = len(input)
    inputColLen = len(input[0])
    for i in range(xOffset, min(windowSize + xOffset, inputRowLen)):
        for j in range(yOffset, min(windowSize + yOffset, inputColLen)):
            if input[i][j] > res:
                res = input[i][j]
    
    return res

def avg(input, windowSize, xOffset, yOffset):
    
    res = 0
    for i in range(xOffset, windowSize + xOffset):
        for j in range(yOffset, windowSize + yOffset):
            res += input[i][j]
    
    return res / (windowSize * windowSize)

def testConvolution():
    inputMatrix = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]

    featureMatrix = [
        [1,0],
        [0,1],
    ]

    convOutput = convolution(inputMatrix, featureMatrix)

    for i in range(len(convOutput)):
        print(convOutput[i])

def testPooling():
    inputMatrix = [
        [1,0.2,1,0.3,1],
        [0.1,0.4,1,0.5,1],
        [0.2,0.6,0.9,0.7,1],
        [0.3,0.8,0.8,0.9,1],
        [0.4,0.9,0.7,0.99,1],
    ]

    output = pool(inputMatrix)

    for i in range(len(output)):
        print(output[i])

def testFullyConnected():
    m1 = [
        [1,2],
        [3,4]
    ]

    m2 = [
        [5,6],
        [7,8]
    ]

    print(fullyConnectedFunction([m1,m2]))

def main():

    # testConvolution()
    # testPooling()
    # testFullyConnected()

    input = [[0 for x in range(16) ] for y in range(16)]
    for i in range(16):
        input[i][i] = 1

    convLayer = ConvolutionLayer()
    activLayer = NeuralLayer(sigmoid)
    poolLayer = NeuralLayer(pool)
    
    layers = [convLayer, activLayer, poolLayer]
    network = Network(layers)
    network.train(input, [1, 0])

if __name__ == '__main__':
    main()