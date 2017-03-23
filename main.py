import math
import cv2
import numpy
import os
import sys
from network.Neuron import Neuron
from network.NeuralLayer import NeuralLayer
from network.CustomLayers import ConvolutionLayer
from network.Network import Network
from network.Matrix import Matrix
from network import constants

def pool(input: Matrix):
    """
    Performs pooling on the input matrix and returns a smaller matrix
    """

    windowSize = 2
    stride = 2

    output = []
    for i in range(0, input.height(), stride):

        outputRow = []
        for j in range(0, input.width(), stride):
            maxVal = max(input, windowSize, i, j)
            outputRow.append(maxVal)

        output.append(outputRow)

    return Matrix.convert(output)

def max(input: Matrix, windowSize, xOffset, yOffset):

    res = input[xOffset][yOffset]
    for i in range(xOffset, min(windowSize + xOffset, input.height())):
        for j in range(yOffset, min(windowSize + yOffset, input.width())):
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


def getImagePixelMatrices():
    dir = "./lane_images/cordova1/"
    files = os.listdir(dir)

    for file in files:
        return getPixelMatrix(dir, file)

def getPixelMatrix(dir, file):
    img = cv2.imread(dir + file)
    if constants.GREYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def main():

    # testConvolution()
    # testPooling()
    # testFullyConnected()

    # Call Open CV here to get the input
    # image = [[0 for x in range(16) ] for y in range(16)]



    # image = Matrix([16, 16])
    # for i in range(16):
    #     image[15 - i][i] = 1
    #
    # input = image
    # print (str(input))

    input = getImagePixelMatrices()
    input = Matrix.convert(input)

    convLayer = ConvolutionLayer(activation = constants.relu)
    # activLayer = NeuralLayer(constants.sigmoid)
    poolLayer = NeuralLayer(func = pool)

    layers = [convLayer, poolLayer]
    network = Network(layers)
    network.train(input, [1, 0])




    test = Matrix([16, 16])
    offset = 1
    # Diagonalise the center
    for i in range(4, 12):
        test[15 - i][i] = 1
    # Off-center the diagonals at the ends
    for i in range(0, 4):
        test[15 - i][i + offset] = 1
    for i in range(12, 16):
        test[15 - i][i - offset] = 1

    # print ("\nTesting Image (Expecting a 'yes'): " + str(test))
    # result = network.run(test)
    # print("CNN Verdict: " + "YES" if result == 1 else "NO")

if __name__ == '__main__':
    main()
