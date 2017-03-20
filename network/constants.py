NEURON_COUNT = 3
EPSILON = 0.0001
LEARNING_RATE = 0.1


def adeline(input, error, weights):
    """
    Performs adeline to get the next set of weights
    """
    if input == None:
        return weights

    output = [[0 for x in range(len(weights))] for y in range(len(weights[0]))]
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            output[i][j] = weights[i][j] + LEARNING_RATE * error * input[i][j]

    return output