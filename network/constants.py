from network.Matrix import Matrix
import math

NEURON_COUNT = 3
EPSILON = 0.1
LEARNING_RATE = 0.1


def adeline(input: Matrix, error, weights: Matrix):
    """
    Performs adeline to get the next set of weights
    """
    if input == None:
        return weights

    # output = [[0 for x in range(len(weights))] for y in range(len(weights[0]))]
    output = Matrix(weights.size())
    for i in range(min(input.height(), weights.height())):
        for j in range(min(input.width(), weights.width())):
            output[i][j] = weights[i][j] + LEARNING_RATE * error * input[i][j]

    return output


def adeline_general(input: Matrix, error, weights: Matrix):
    """
    Performs adeline to get the next set of weights
    """
    if input == None:
        return weights

    output = Matrix(weights.size())
    for i in range(min(input.height(), weights.height())):
        for j in range(min(input.width(), weights.width())):
            output[i][j] = weights[i][j] + LEARNING_RATE * error * input[i][j]

    return output

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

def relu(input: Matrix):
    """
    Sets all negative numbers to 0. Returns a matrix of the same size
    """
    output = Matrix(input.size())

    # Perform sigmoid on all elements in input matrix
    for x in range(input.height()):
        for y in range(input.width()):
            output[x][y] = input[x][y] if input[x][y] > 0 else 0 

    return output

