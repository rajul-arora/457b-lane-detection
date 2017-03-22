from network.Matrix import Matrix

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