import math
import cv2
import numpy as np
import os
import sys
from network.Neuron import Neuron
from network.NeuralLayer import NeuralLayer
from network.CustomLayers import ConvolutionLayer
from network.Network import Network
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD
from network.Matrix import Matrix
from network import constants
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Dense, Activation
from keras.applications.vgg19 import VGG19
import itertools

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


def getInputAndOutputMatrices():
    dir = "./lane_images/cordova1_images/"
    files = os.listdir(dir)

    for file in files:
        yield getPixelMatrices(dir, file)

def generateTrainTestDataSets():
    inputOutputIterator = getInputAndOutputMatrices()
    train = []
    test = []

    for img, output_matrix in inputOutputIterator:
        train.append(img)
        test.append(output_matrix)
    return (train, test)

def getPixelMatrices(dir, file):
    img = cv2.imread(dir + file)
    output_filename = file.split('.')[0] + 'matrix.txt'
    if constants.GREYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_matrix = np.fromfile(constants.OUTPUT_DIRECTORY + output_filename, sep=" ").reshape(constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH)
    return img, output_matrix


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


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

    datasetIterator = getInputAndOutputMatrices()

    print("[INFO]: Generating Input Feature Models.")

    # for i in range(0,10):
        # t = train[i]

    model = VGG19(weights='imagenet', include_top=False)
    x_train = []
    x_test = []
    shape = None
    for img, test in itertools.islice(datasetIterator, 10):
        res = model.predict(np.array([img]))[0]
        x_train.append(res)
        x_test.append(test)
        shape = res.shape

    print(shape)
    npArray = np.array(x_test)
    testData = npArray.transpose().swapaxes(0, 1)
    input = Input(shape=shape)

    x = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_last')(input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', data_format='channels_last')(x)
    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    result = autoencoder.fit(np.array(x_train), testData, epochs=10)

        # network = Sequential()
    # network.add(Conv2D(32, (16, 16), activation='relu', padding='same'))
    # network.add(UpSampling2D((2, 2)))
    # network.add(Conv2D(32, (16, 16), activation='relu', padding='same'))
    # network.add(UpSampling2D((2, 2)))
    # network.add(Conv2D(64, (16, 16), activation='relu', padding='same'))
    # network.add(UpSampling2D((2, 2)))
    # network.add(Conv2D(1, (16, 16), activation='sigmoid', padding='same'))
    #
    # network.compile(loss='mean_squared_error', optimizer='adadelta')
    # result = network.fit(np.array(x_train), np.array(x_test), epochs=10)


    # for t in train:
    #     image = np.expand_dims(t, axis=0)
    #     x_train.append(model.predict(image))


    # inputOutputIterator = getInputAndOutputMatrices()
    # partialInputOutputMatrices = []
    # for wholeImg, output_matrix in inputOutputIterator:
        # partialInputImages = blockshaped(wholeImg, constants.PARTIAL_IN_IMG_DIM[0], constants.PARTIAL_IN_IMG_DIM[1])
        # partialOutputMatrices = blockshaped(output_matrix, constants.PARTIAL_IN_IMG_DIM[0], constants.PARTIAL_IN_IMG_DIM[1])
        # for i in range(len(partialInputImages)):
        #     partialInputOutputMatrices.append((Matrix.convert(partialInputImages[i]), Matrix.convert(partialOutputMatrices[i])))

    # convLayer = ConvolutionLayer(activation = constants.relu)
    # # activLayer = NeuralLayer(constants.sigmoid)
    # poolLayer = NeuralLayer(func = pool)
    #
    # layers = [convLayer, poolLayer]
    # network = Network(layers)
    #network.train(partialInputImagesAsMatrices[0], [1, 0]) #Just try the first input image segment for now



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
