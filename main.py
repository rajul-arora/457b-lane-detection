import math
import uuid

EPOCH_COUNT = 50
TRAINING_SIZE = 400
BATCH_SEP = 0.75

import cv2
import numpy as np
import os
import sys
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
import uuid
from keras.optimizers import SGD
from network.Matrix import Matrix
from network import constants
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Dense, Activation
from keras.applications.vgg19 import VGG19
import itertools

def getInputAndOutputMatrices():
    dir = "./lane_images/cordova2_images/"
    files = os.listdir(dir)

    for file in files:
        yield getPixelMatrices(dir, file)

def normalize(x):
    x_mins = np.min(x, axis=0)
    x_maxs = np.max(x, axis=0)
    x_deltas = x_maxs - x_mins
    return (x - x_mins) / x_deltas

def generateTrainTestDataSets():
    inputOutputIterator = getInputAndOutputMatrices()
    train = []
    test = []

    for img, output_matrix in itertools.islice(inputOutputIterator, TRAINING_SIZE):
        nImg = normalize(img)
        train.append(np.expand_dims(nImg, axis=2))
        test.append(np.expand_dims(output_matrix, axis=2))
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

def segnet(inputImage, filename, shouldSaveWeights=True, shouldTrain=False, train_data=None, test_data=None):
    print("Entering Segnet")
    windowSize = (3, 3)
    inputShape = Input(shape=(480, 640, 1))

    x = Conv2D(32, windowSize, activation='relu', padding='same')(inputShape)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, windowSize, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, windowSize, activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, windowSize, activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, windowSize, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, windowSize, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, windowSize, activation='sigmoid', padding='same')(x)
    autoencoder = Model(inputShape, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    if os.path.isfile(filename):
        print("Reading File")
        autoencoder.load_weights(filename)

    if shouldTrain and test_data is not None and train_data is not None:
        print("Training test data")
        autoencoder.fit(np.array(train_data), np.array(test_data), epochs=EPOCH_COUNT)

    if shouldSaveWeights:
        autoencoder.save_weights(filename)
        print("Saved Weights")

    guess = autoencoder.predict(np.expand_dims(inputImage, axis=0))
    return guess

def main():
    (train_data, test_data) = generateTrainTestDataSets()
    x_train = train_data[:TRAINING_SIZE * BATCH_SEP]
    x_test = test_data[:TRAINING_SIZE * BATCH_SEP]
    verify = train_data[TRAINING_SIZE * BATCH_SEP : TRAINING_SIZE]

    guess = segnet(verify, "weights.h5", shouldTrain=True, test_data=x_test, train_data=x_train)

    for g in guess:
        cvMatrix = [[[] for i in range(0, len(guess[0][0]))] for j in range(0, len(guess[0]))]
        for i in range(0,len(cvMatrix)):
            for j in range(0, len(cvMatrix[0])):
                cvMatrix[i][j].append(int(255 * g[i][j]))
                cvMatrix[i][j].append(int(255 * g[i][j]))
                cvMatrix[i][j].append(int(255 * g[i][j]))

        cv2.imwrite('img/testMono-' + str(uuid.uuid4()) + '.jpg', np.array(cvMatrix))

if __name__ == '__main__':
    main()
