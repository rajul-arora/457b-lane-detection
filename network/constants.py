from network.Matrix import Matrix
import math

NEURON_COUNT = 3
EPSILON = 0.1
LEARNING_RATE = 0.1
GREYSCALE = True
PARTIAL_IN_IMG_DIM = [32, 32]
OUTPUT_DIRECTORY = './lane_images/cordova1_output_matrices/'
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

def sigmoid(input: Matrix):
    """
    Performs sigmoid on all elements in input and returns a matrix of the same size
    """

    # Instantiate output as a matrix same dimensions as input
    # output = [ [0 for i in range(len(input))] for j in range(len(input[0])) ] 
    output = Matrix(input.size())

    # Perform sigmoid on all elements in input matrix
    for x in range(input.height()):
        for y in range(input.width()):
            output[x][y] = 1 /  (1 + math.exp(-1 * input[x][y])) 

    return output

def sigderiv(input):    
    print(input)
    return (-1 * math.exp(-1 * input)) / (1 + math.exp(-1 * input)) ** 2

def ReLU(input: Matrix):
    """
    Sets all negative numbers to 0. Returns a matrix of the same size
    """
    output = Matrix(input.size())

    # Perform sigmoid on all elements in input matrix
    for x in range(input.height()):
        for y in range(input.width()):
            output[x][y] = input[x][y] if input[x][y] > 0 else 0 

    return output

def dReLU(X: Matrix):
    """
    Calculates the derivative of ReLU of the given input matrix X.
    """
    output = Matrix(X.size())

    for i in range(X.height()):
        for j in range(X.width()):
            output[i][j] = 1 if X[i][j] > 0 else 0

    return output
